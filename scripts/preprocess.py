import argparse
import os
import torch.multiprocessing as mp

from detectron2.config import get_cfg
from detectron2.engine import default_setup

from context_rcnn.config import add_context_rcnn_config
from context_rcnn.preprocess import build_banks


# relative to models-dir
_MODELS = {
    "frcnn-r101-cct": {
        "weights": "pretrained/cct-animal-frcnn-r101/model_final_wo_solver_states.pth",
        "config": "pretrained/cct-animal-frcnn-r101/config.yaml"
    },
}

def get_model_for_inference(model_name, models_dir, score_threshold=0.05, nms_threshold=0.5):
    weights = os.path.join(models_dir, _MODELS[model_name]["weights"])
    config = os.path.join(models_dir, _MODELS[model_name]["config"])
    
    cfg = get_cfg()
    add_context_rcnn_config(cfg)
    
    if config is not None:
        cfg.merge_from_file(config)
        
    if weights is not None:
        cfg.MODEL.WEIGHTS = weights
        
    # the configurable part
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
    default_setup(cfg, {})
    
    return cfg

def preprocess_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="frcnn-r101-cct", help="Name of model to use for inference. See _MODELS")
    parser.add_argument("--dataset", default="cct", help="Name of dataset to generate banks for, { 'cct' }")
    parser.add_argument("--banks-dir", default="./banks", metavar="FILE", help="Location of memory banks for Context R-CNN")
    parser.add_argument("--data-dir", default="../data", metavar="FILE", help="Path to data/")
    parser.add_argument("--models-dir", default="../models", metavar="FILE", help="Path to models/")
    parser.add_argument("--num-gpus", type=int, default=1)
    return parser

def build_banks_with_process(gpu_idx, cfg, dataset_name, data_dir, bank_dir, num_gpus):
    """Same as context_rcnn.preprocess but ignores process ID from torch multiprocessing."""
    cfg = cfg.clone()
    cfg.MODEL.DEVICE = "cuda:{}".format(gpu_idx)
    print("Launching process on GPU", gpu_idx)
    build_banks(cfg, dataset_name, data_dir, bank_dir, gpu_idx, num_gpus)

if __name__ == "__main__":
    args = preprocess_argument_parser().parse_args()
    cfg = get_model_for_inference(args.model, args.models_dir, score_threshold=0.0)
    mp.spawn(build_banks_with_process, args=(cfg, args.dataset, args.data_dir, args.banks_dir, args.num_gpus), 
            nprocs=args.num_gpus)