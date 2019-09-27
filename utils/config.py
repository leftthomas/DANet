import torch

cfg = dict(EPOCH=0, CLASS_UNIFORM_PCT=0, BATCH_WEIGHTING=False, BORDER_WINDOW=1, REDUCE_BORDER_EPOCH=-1,
           STRICTBORDERCLASS=None,
           DATASET=dict(CITYSCAPES_DIR='/home/data/cityscapes', CV_SPLITS=3),
           MODEL=dict(BN='regularnorm', BNFUNC=torch.nn.BatchNorm2d, BIGMEMORY=False))
