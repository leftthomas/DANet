# GSCNN
This is the official code for [Gated-SCNN: Gated Shape CNNs for Semantic Segmentation](https://arxiv.org/abs/1907.05740).

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* torch-encoding
```
pip install torch-encoding
```
* tensorboard
```
pip install tb-nightly
```
- opencv
```
conda install opencv
```

## Download pretrained models
Download the pretrained model from the [Google Drive Folder](https://drive.google.com/file/d/1wlhAXg-PfoUM-rFy2cksk43Ng3PpsK2c/view), 
and save it in `checkpoints`.

## Evaluation (Cityscapes)
```bash
python train.py --evaluate --snapshot checkpoints/best_cityscapes_checkpoint.pth
```

