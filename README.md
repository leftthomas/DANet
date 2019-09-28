# GSCNN
This is the official code for [Gated-SCNN: Gated Shape CNNs for Semantic Segmentation](https://arxiv.org/abs/1907.05740).

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* tensorboard
```
pip install tb-nightly
```
- mmcv
```
pip install mmcv
```
- mmdetection
```
python setup.py develop
```

## Prepare datasets

It is recommended to symlink the dataset root to `$MMDETECTION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```
The cityscapes annotations have to be converted into the coco format using the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) toolbox.
We plan to provide an easy to use conversion script. For the moment we recommend following the instructions provided in the 
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/data) toolbox. When using this script all images have to be moved into the same folder. On linux systems this can e.g. be done for the train images with:
```shell
cd data/cityscapes/
mv train/*/* train/
```

## Scripts

[Here](https://gist.github.com/hellock/bf23cd7348c727d69d48682cb6909047) is
a script for setting up mmdetection with conda.
