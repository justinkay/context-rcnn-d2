import argparse
import os
import torch.multiprocessing as mp

from detectron2.config import get_cfg
from detectron2.engine import default_setup

from context_rcnn.config import add_context_rcnn_config
from context_rcnn.preprocess import build_banks, build_one_bank


# relative to models-dir
_MODELS = {
    # 1-class model
    "frcnn-r101-cct": {
        "weights": "pretrained/cct-animal-frcnn-r101/model_final_wo_solver_states.pth",
        "config": "pretrained/cct-animal-frcnn-r101/config.yaml"
    },
    
    # toy model undertrained on just toy1924-species
    "toy": {
        "weights": "toy1924-multi-18epoch/model_final.pth",
        "config": "toy1924-multi-18epoch/config.yaml"
    },
    
    # model straight from detectron
    "c4-d2": {
        "weights": "pretrained/d2/model_final_298dad.pkl",
        "config": "pretrained/d2/config.yaml"
    },
}

def get_cfg_for_inference(model_name, models_dir, score_threshold=0.0, nms_threshold=0.5):
    weights = os.path.join(models_dir, _MODELS[model_name]["weights"])
    config = os.path.join(models_dir, _MODELS[model_name]["config"])
    
    cfg = get_cfg()
    add_context_rcnn_config(cfg)
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
    
    return cfg

def preprocess_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="frcnn-r101-cct", help="Name of model to use for inference. See _MODELS")
    parser.add_argument("--dataset", default="cct", help="Name of dataset to generate banks for, { 'cct' }")
    parser.add_argument("--banks-dir", default="./banks", metavar="FILE", help="Location of memory banks for Context R-CNN")
    parser.add_argument("--data-dir", default="../data", metavar="FILE", help="Path to data/")
    parser.add_argument("--models-dir", default="../models", metavar="FILE", help="Path to models/")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--only", default=None, help="only do the specified bank")
    return parser

def build_banks_with_process(gpu_idx, cfg, dataset_name, data_dir, bank_dir, num_gpus):
    """Same as context_rcnn.preprocess but ignores process ID from torch multiprocessing."""
    cfg = cfg.clone()
    cfg.MODEL.DEVICE = "cuda:{}".format(gpu_idx)
    default_setup(cfg, {})
    
    print("Launching process on GPU", gpu_idx)
    build_banks(cfg, dataset_name, data_dir, bank_dir, gpu_idx, num_gpus)

if __name__ == "__main__":
    args = preprocess_argument_parser().parse_args()
    cfg = get_cfg_for_inference(args.model, args.models_dir)
    
    if args.only:
        build_one_bank(cfg, args.dataset, args.data_dir, args.banks_dir, args.only, in_train=True)
    else:
        mp.spawn(build_banks_with_process, args=(cfg, args.dataset, args.data_dir, args.banks_dir, args.num_gpus), 
                nprocs=args.num_gpus)