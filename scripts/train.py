import logging
import os
from collections import OrderedDict
import torch

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import default_setup, DefaultTrainer, hooks, default_argument_parser, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from context_rcnn.config import add_context_rcnn_config
from context_rcnn.data import register_dataset
from context_rcnn.train import get_trainer as get_context_trainer
# to register architectures
import context_rcnn.rcnn 
import context_rcnn.roi_heads


_MODELS = {
    "frcnn-r101": { "weights": "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl",
                     "config": "../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" },
    "context-rcnn-r101": { "weights": "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl",
                     "config": "../configs/COCO-Detection/context_rcnn_R_101_FPN.yaml" },
}

def get_training_config(data_dir="../data", model="context-rcnn-r101", device="cuda", num_gpus=8, banks_dir="./banks", dataset="cct", 
                        subset="default", weights_path=None, lr=None, temp=0.01, amp=False, freeze=False):
    custom_weights_supplied = weights_path is not None
    weights_path = weights_path or _MODELS[model]["weights"]
    config_path = _MODELS[model]["config"]

    cfg = get_cfg()
    
    # perform bs / lr scaling based on num gpus
    if amp:
        bs = num_gpus * 4
    else:
        bs = num_gpus * 2
    lr = 0.005 * bs / 16 # Detectron2 default is 0.02 * bs / 16
        
    add_context_rcnn_config(cfg)
    
    # do this first, because we will overwrite
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    
    # enable mixed precision training
    cfg.SOLVER.AMP.ENABLED = amp
    
    # this has worked well for me on AWS p3; YMMV
    # if you get a bunch of OpenBLAS errors during evaluation, try reducing this
    cfg.DATALOADER.NUM_WORKERS = int(max(2, num_gpus))
    
    cfg.MODEL.DEVICE = device
    cfg.SOLVER.IMS_PER_BATCH = bs
    cfg.SOLVER.BASE_LR = lr
    
    # use a seed for reproducibility
    cfg.SEED = 42
    
    # set up data
    train_key, val_key, num_classes = register_dataset(data_dir, dataset, subset)
    cfg.DATASETS.TRAIN = (train_key,)
    cfg.DATASETS.TEST = (val_key,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    num_imgs = len(DatasetCatalog.get(train_key))
    epoch_size = int(num_imgs / bs)
    
    # based on Detectron2 defaults, which trained COCO for ~37 epochs
#     cfg.SOLVER.MAX_ITER = 37*epoch_size
#     cfg.SOLVER.STEPS = (24*epoch_size, 32*epoch_size)
    
    # or try a reduced schedule
    cfg.SOLVER.MAX_ITER = 18*epoch_size
    cfg.SOLVER.STEPS = (12*epoch_size, 16*epoch_size)
    
    # make sure warmup period is shorter than first training stage
    if cfg.SOLVER.WARMUP_ITERS >= cfg.SOLVER.STEPS[0]:
        cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.STEPS[0]/2)

    # evaluate and save after every epoch
    # makes this a multiple of PeriodicWriter default period so that eval metrics always get written to 
    # Tensorboard / WandB
    approx_epoch = max(20, epoch_size // 20 * 20)
    cfg.TEST.EVAL_PERIOD = approx_epoch
    cfg.SOLVER.CHECKPOINT_PERIOD = approx_epoch
    cfg.VIS_PERIOD = approx_epoch
    
    # little hack for testing
#     if "month" in subset:
#         cfg.TEST.EVAL_PERIOD = 1e10
#         cfg.SOLVER.CHECKPOINT_PERIOD = 1e10
    
    cfg.MODEL.CONTEXT.BANKS_DIR = banks_dir
    cfg.MODEL.CONTEXT.SOFTMAX_TEMP = temp
    
    # name output directory after model name
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_" + model + "_" + dataset + "_" + subset
    if custom_weights_supplied:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_pretrained"
    if freeze:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_freeze"
        
    # run some default setup from Detectron2
    # note: this eliminates:
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, {})

    return cfg

def get_coco_trainer(cfg, resume=False):
    trainer = COCOTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer    

class COCOTrainer(DefaultTrainer):
    """
    A basic Trainer with a COCOEvaluator, set up for distributed training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, distributed=True)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def training_argument_parser():
    # get parse with Detectron2 default commands
    parser = default_argument_parser()

    parser.add_argument("--model", default="context-rcnn-r101", help="name of model to train, { 'frcnn-r101', 'context-rcnn-r101' }")
    parser.add_argument("--data-dir", default="../data", metavar="FILE", help="path to data/")
    parser.add_argument("--device", default="cuda", help="{ 'cuda', 'cpu' }")
    parser.add_argument("--wandb", help="name of wandb project to log tensorboard results")
    parser.add_argument("--banks", default="./banks", help="Location of memory banks for Context R-CNN")
    parser.add_argument("--dataset", default="cct", help="{ 'cct' }")
    parser.add_argument("--subset", default="default", help="{ 'default' }")
    parser.add_argument("--weights", default=None, help="path to pth file for alternate weights initialization. used for models with alternate (non-COCO) pretraining, etc.")
    parser.add_argument("--lr", default=None, type=float, help="custom learning rate; does not scale with num GPUs")
    parser.add_argument("--temp", default=0.01, type=float, help="softmax temperature for attention network")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision training with torch native AMP")
    parser.add_argument("--freeze", action="store_true", help="freeze everything except the long_term context module")
    
    return parser

def main(args):
    if args.wandb:
        import wandb; wandb.init(project=args.wandb, sync_tensorboard=True)
        
    cfg = get_training_config(model=args.model, data_dir=args.data_dir, device=args.device, num_gpus=args.num_gpus,
                              banks_dir=args.banks, dataset=args.dataset, subset=args.subset, 
                              weights_path=args.weights, lr=args.lr, temp=args.temp, amp=args.amp, freeze=args.freeze)
    
    if "context" in args.model:
        trainer = get_context_trainer(cfg, args.resume, args.freeze)
    else:
        trainer = get_coco_trainer(cfg, args.resume)
        
    return trainer.train()

if __name__ == "__main__":
    args = training_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
