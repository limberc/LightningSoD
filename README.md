# FastSoD

This repo has been heavily inherit from this [repo](https://github.com/yun-liu/FastSaliency).
We have modified this repo based on PyTorch Lightning.

The data folder structure is presented as below:

```bash
data
├── DUT-OMRON
│   ├── DUT-OMRON-bounding-box
│   ├── DUT-OMRON-eye-fixations
│   ├── images
│   └── masks
└── ECSSD
    ├── images
    └──  masks
```

## Training

`python lightning.py fit --model.model {SAMNet OR HVPNet}`

## Infer

`python lightning.py test --model.model {SAMNet OR HVPNet}`
