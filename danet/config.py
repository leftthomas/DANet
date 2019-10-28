from detectron2.config import CfgNode as CN


def add_danet_config(cfg):
    """
    Add config for DANet.
    """
    _C = cfg

    _C.MODEL.DILATED_RESNET = CN()

    _C.MODEL.DILATED_RESNET.DEPTH = 50
    _C.MODEL.DILATED_RESNET.OUT_FEATURES = ["res4"]
    _C.MODEL.DILATED_RESNET.NORM = "FrozenBN"
    _C.MODEL.DILATED_RESNET.WIDTH_PER_GROUP = 64
    _C.MODEL.DILATED_RESNET.STRIDE_IN_1X1 = True
