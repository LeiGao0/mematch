# mematch
[![DOI](https://zenodo.org/badge/1195970146.svg)](https://doi.org/10.5281/zenodo.19343638)

This code is associated with the manuscript submitted to The Visual Computer.If you use this code, please cite our corresponding paper.

## Getting Started
### Installation

```bash
cd MeMatch
conda create -n MeMatch python=3.10.4
conda activate MeMatch
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
### Pre-trained Encoders

[DINOv2-Small](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | [DINOv2-Base](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | [DINOv2-Large](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)

```
├── ./pretrained
    ├── dinov2_small.pth
    ├── dinov2_base.pth
    └── dinov2_large.pth
```

### Datasets
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)
Please modify your dataset path in configuration files.
- RailSem19: [RailSem19]( www.wilddash.cc)
Please modify your dataset path in configuration files.

## Training

### MeMatch
```bash
python mematch.py
```
### FixMatch
```bash
python fixmatch.py
```
### Supervised Baseline
```bash
python unimatch_v2.py 
```
