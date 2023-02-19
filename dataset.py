import os.path as osp
from glob import glob

from PIL import Image
from torch.utils.data import Dataset


class SaliencyDataset(Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, data_dir, dataset, transform=None):
        '''
        :param data_dir: directory where the dataset is kept
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.data_dir = data_dir
        self.transform = transform
        self.img_list = sorted(glob(osp.join(data_dir, dataset, 'images', '*.jpg')))
        self.msk_list = sorted(glob(osp.join(data_dir, dataset, 'masks', '*.png')))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image = Image.open(self.img_list[idx]).convert('RGB')
        label = Image.open(self.msk_list[idx]).convert('L')
        if self.transform is not None:
            [image, label] = self.transform(image, label)

        return image, label
