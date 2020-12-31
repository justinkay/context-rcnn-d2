from detectron2.config import CfgNode as CN


def add_context_rcnn_config(cfg):
    """Add Context R-CNN config values to an existing CfgNode."""
    _C = cfg

    _C.MODEL.CONTEXT = CN()
    
    # defaults from paper
    _C.MODEL.CONTEXT.NUM_CONTEXT_ITEMS = 8500
    _C.MODEL.CONTEXT.D1 = 2048 # hidden layer 1
    _C.MODEL.CONTEXT.D2 = 2048 # hidden layer 2
    _C.MODEL.CONTEXT.SOFTMAX_TEMP = 0.01
    _C.MODEL.CONTEXT.NUM_INPUT_FEATS = 2048
    _C.MODEL.CONTEXT.NUM_CONTEXT_FEATS = 2048

    # settings for FPN w/o datetime (TODO: put into different config file)
    _C.MODEL.CONTEXT.NUM_INPUT_FEATS = 256
    _C.MODEL.CONTEXT.NUM_CONTEXT_FEATS = 260
    _C.MODEL.CONTEXT.NUM_CONTEXT_ITEMS = 2000
    _C.MODEL.CONTEXT.D1 = 256
    _C.MODEL.CONTEXT.D2 = 256
    
    _C.MODEL.CONTEXT.BANKS_DIR = './banks'