# DANet
A PyTorch implementation of DANet based on CVPR 2019 paper [Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983). 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
- opencv
```
pip install opencv-python
```
- tensorboard
```
pip install tensorboard
```
- pycocotools
```
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```
- fvcore
```
pip install git+https://github.com/facebookresearch/fvcore
```
- cityscapesScripts
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```
- detectron2
```
pip install git+https://github.com/facebookresearch/detectron2.git@master
```

## Datasets
For a few datasets that detectron2 natively supports, the datasets are assumed to exist in a directory called
`datasets/`, under the directory where you launch the program. They need to have the following directory structure:

### Expected dataset structure for Cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
run `./datasets/prepare_cityscapes.py` to creat `labelTrainIds.png`.

## Training
To train a model, run
```bash
python train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end DANet training with ResNet-50 backbone on 8 GPUs, one should execute:
```bash
python train_net.py --config-file configs/r50.yaml --num-gpus 8
```

## Evaluation
Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/r50.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS checkpoints/model.pth
```

## Results
There are some difference between this implementation and official implementation:
1. No `Multi-Grid` and `Multi-Scale Testing`;
2. The image sizes of `Multi-Scale Training` are (800, 832, 864, 896, 928, 960);
3. Training step is set to `24000`;
4. Learning rate policy is `WarmupMultiStepLR`;
5. `Position Attention Module (PAM)` uses the similar mechanism as `Channel Attention Module (CAM)`, just uses the tensor
and its transpose to compute attention. 

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Name</th>
		<th>train time (s/iter)</th>
		<th>inference time (s/im)</th>
		<th>train mem (GB)</th>
		<th>PA</br>%</th>
		<th>mean PA %</th>
		<th>mean IoU %</th>
		<th>FW IoU %</th>
		<th>download link</th>
		<!-- TABLE BODY -->
		<!-- ROW: r50 -->
		<tr>
			<td align="center"><a href="configs/r50.yaml">R50</a></td>
			<td align="center">0.49</td>
			<td align="center">0.12</td>
			<td align="center">27.12</td>
			<td align="center">94.19</td>
			<td align="center">75.31</td>
			<td align="center">66.64</td>
			<td align="center">89.54</td>
			<td align="center"><a href="https://pan.baidu.com/s/18wRQbLQyqXA4ISloUGWTSA">model</a>&nbsp;|&nbsp;ga7k</td>
		</tr>
		<!-- ROW: r101 -->
		<tr>
			<td align="center"><a href="configs/r101.yaml">R101</a></td>
			<td align="center">0.315</td>
			<td align="center">0.102</td>
			<td align="center">5.0</td>
			<td align="center">53.6</td>
			<td align="center">0.102</td>
			<td align="center">5.0</td>
			<td align="center">53.6</td>
			<td align="center"><a href="https://pan.baidu.com/s/18wRQbLQyqXA4ISloUGWTSA">model</a>&nbsp;|&nbsp;ga7k</td>
		</tr>
	</tbody>
</table>