import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError

import transforms
from dataset import SaliencyDataset
from load_data import LoadData
from models import HVPNet, SAMNet

torch.set_float32_matmul_precision("high")


class CrossEntropyLoss(nn.Module):
    def forward(self, inputs, target):
        if isinstance(target, tuple):
            target = target[0]
        target = target.float()
        loss = F.binary_cross_entropy(inputs[:, 0, :, :], target)
        for i in range(1, inputs.shape[1]):
            loss += 0.4 * F.binary_cross_entropy(inputs[:, i, :, :], target)
        return loss


class FastSaliencyDataModule(LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=20, width=336, cached_data_file='dut-omron_train.p',
                 subset="DUT-OMRON"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.width = width
        self.cached_data_file = cached_data_file
        self.subset = subset

    def setup(self, stage: str = None) -> None:
        if not os.path.isfile(self.cached_data_file):
            loader = LoadData(self.data_dir, self.subset, self.cached_data_file)
            self.data = loader.process()
        else:
            self.data = pickle.load(open(self.cached_data_file, 'rb'))
        # ImageNet mean, std
        mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.])
        std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.])

        val_transform = transforms.Compose([
            transforms.Normalize(mean=self.data['mean'], std=self.data['std']),
            transforms.Scale(self.width, self.width),
            transforms.ToTensor()
        ])

        self.train_dataset = {
            "scale1": SaliencyDataset(self.data_dir, self.subset, transform=self.scale_train_transform(2.0, 28.)),
            "scale2": SaliencyDataset(self.data_dir, self.subset, transform=self.scale_train_transform(1.5, 22.)),
            "scale3": SaliencyDataset(self.data_dir, self.subset, transform=self.scale_train_transform(1.25, 22.)),
            "scale4": SaliencyDataset(self.data_dir, self.subset, transform=self.scale_train_transform(.75, 7.)),
            "normal": SaliencyDataset(self.data_dir, self.subset, transform=self.scale_train_transform()),
        }
        self.val_dataset = SaliencyDataset(self.data_dir, "ECSSD", transform=val_transform)

    def get_train_loader(self, scale):
        return DataLoader(self.train_dataset[scale], batch_size=self.batch_size, shuffle=True, num_workers=32,
                          pin_memory=True, drop_last=True)

    def train_dataloader(self):
        return {
            "scale1": self.get_train_loader("scale1"),
            "scale2": self.get_train_loader("scale2"),
            "scale3": self.get_train_loader("scale3"),
            "scale4": self.get_train_loader("scale4"),
            "normal": self.get_train_loader("normal"),
        }

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12,
                          pin_memory=True)

    def scale_train_transform(self, scale_factor: float = 1.0, crop_size: float = 7.):
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=self.data['mean'], std=self.data['std']),
            transforms.Scale(int(self.width * scale_factor), int(self.width * scale_factor)),
            transforms.RandomCropResize(int(crop_size / 224. * self.width)),
            transforms.RandomFlip(),
            transforms.ToTensor()
        ])


class FastSaliencyModule(LightningModule):
    def __init__(self, model: str = 'SAMNet', batch_size=16,
                 pretrained_path='pretrained_model/SAMNet_backbone_pretrain.pth',
                 milestones=[30, 60, 90]):
        super().__init__()
        self.save_hyperparameters()
        self.model = {
            "HVPNet": HVPNet,
            "SAMNet": SAMNet
        }[model](pretrained=pretrained_path)
        self.loss = CrossEntropyLoss()
        self.milestones = milestones
        self.batch_size = batch_size
        # I want to disable auto backward
        self.automatic_optimization = False
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        # Originally, they pass each scale sequentially.
        # I think this should be fine if I process each batch all together in one training step.
        for scale in ['scale1', 'scale2', 'scale3', 'scale4', 'normal']:
            input, target = batch[scale]
            opt = self.optimizers()
            opt.zero_grad()
            output = self(input)
            loss = self.loss(output, target)
            self.log(f"train/loss_{scale}", loss, logger=True, on_step=True, on_epoch=True,
                     add_dataloader_idx=False, batch_size=self.batch_size, sync_dist=True)
            if scale == 'normal':
                self.train_mae(output[:, 0, :, :], target)
                self.log("train/mae", self.train_mae, logger=True, on_step=True, on_epoch=True, sync_dist=True,
                         batch_size=self.batch_size)
            self.manual_backward(loss)
            opt.step()

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        loss = self.loss(output, target)
        self.log("val/loss", loss, logger=True, on_step=True, on_epoch=True, prog_bar=True,
                 add_dataloader_idx=False, batch_size=self.batch_size, sync_dist=True)
        self.val_mae(output[:, 0, :, :], target)
        self.log("val/mae", self.val_mae, logger=True, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    logger = WandbLogger(project='FastSaliencyBaseline', name='baseline',
                         log_model=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    LightningCLI(FastSaliencyModule, FastSaliencyDataModule, save_config_overwrite=True,
                 seed_everything_default=42,
                 trainer_defaults={
                     "max_epochs": 50,
                     "accelerator": "auto",
                     "strategy": "ddp_find_unused_parameters_false",
                     "devices": 4,
                     "logger": logger,
                     "benchmark": True,
                     "callbacks": [lr_monitor]
                 })
