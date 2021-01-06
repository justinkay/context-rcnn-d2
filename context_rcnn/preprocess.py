import argparse
import cv2
import glob
import os
import pandas as pd
import pickle
import torch
from tqdm import tqdm

from detectron2.data import DatasetCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads import Res5ROIHeads, StandardROIHeads

from context_rcnn.data import register_dataset, _DATASETS


class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out
        
def get_feats_for_im(predictor, im):
    """Args:
        predictor: a detectron2.engine.defaults.DefaultPredictor
        im: an image loaded with cv2.imread(...)
    """
    roi_heads = predictor.model.roi_heads
    
    # register a hook for this image to keep intermediate output from backbone
    roi_heads_io = SaveIO()
    handle = roi_heads.register_forward_hook(roi_heads_io)
    
    # run inference and grab backbone features
    output = predictor(im)
    images, features, proposals, gt_instances = roi_heads_io.input

    # transform predictions back into img dimensions seen by model
    tfm = predictor.aug.get_transform(im)
    instances = detector_postprocess(output["instances"], output_height=tfm.new_h, output_width=tfm.new_w)
    
    # TODO support more than 1 box
    top_pred = instances.pred_boxes[0]
    
    # get instance features
    feats_list = [features[f] for f in roi_heads.in_features]
    if isinstance(roi_heads, Res5ROIHeads):
        box_features = roi_heads._shared_roi_transform(feats_list, [top_pred]) # -> num_proposals x 2048 x 7 x 7
        # pool over spatial dimensions to get final feats
        box_features = box_features.mean([-2, -1])                             # -> num_proposals x 2048
    elif isinstance(roi_heads, StandardROIHeads):
        box_features = roi_heads.box_pooler(feats_list, [top_pred])
        box_features = roi_heads.box_head(box_features)                        # -> num_proposals x 1024
    else:
        print("ROI heads class not supported:", type(roi_heads))
        
    # TODO assumes one proposal
    final_feats = torch.squeeze(box_features)
        
    # add scaled bbox info as additional feats
    img_wh = torch.Tensor([tfm.new_w, tfm.new_h]).to(top_pred.tensor.device)
    centers = torch.div(top_pred.get_centers()[0], img_wh)
    box = top_pred.tensor
    bbox_w = (box[0, 2] - box[0, 0])
    bbox_h = (box[0, 3] - box[0, 1])
    bbox_wh = torch.div(torch.Tensor((bbox_w, bbox_h)).to(final_feats.device), img_wh)
    embedding = torch.cat((final_feats, centers, bbox_wh))
    
    # de-register the hook
    handle.remove()
    
    return embedding
    
def build_banks(cfg, dataset_name, data_dir, bank_dir, gpu_idx=0, num_gpus=1):
    """
    Args:
        cfg: Detectron2 config for model to be used for inference
        dataset_name: key from context_rcnn.data._DATASETS
        data_dir: location of data/
        bank_dir: output location for feature banks
        gpu_idx: which gpu this method is to be run on
        num_gpus: total number of gpus running inference
    """
    if not os.path.exists(bank_dir):
        os.mkdir(bank_dir)
        
    predictor = DefaultPredictor(cfg)
    register_dataset(data_dir, dataset_name)
    img_loc = os.path.join(data_dir, _DATASETS[dataset_name]["imgs_loc"])
    
    for dataset in (dataset_name + "_train", dataset_name + "_val"):
        print("Building banks for locations in", dataset)
        dataset_dict = DatasetCatalog.get(dataset)
        df = pd.DataFrame(dataset_dict)
        
        # filter by locations intended for this gpu
        # TODO should split this better for imbalanced data
        locs = df.location.unique()
        locs_per_gpu = len(locs) // num_gpus
        start_ind = (gpu_idx)*locs_per_gpu
        if gpu_idx == (num_gpus-1):
            # include any leftovers if fractional num locs per gpu
            end_ind = len(locs)
        else:
            end_ind = (gpu_idx+1)*locs_per_gpu
        my_locs = locs[start_ind:end_ind]
        df = df[df.location.isin(my_locs)]
        
        by_location = df.groupby("location")
        for i, (location, df_group) in enumerate(by_location):
            # dicts mapping img_id: feats (TODO key by datetime)
            output = {} 
            flipped_output = {}
            im_ids = df_group.image_id.values
            for im_id in tqdm(im_ids, desc="Location {}; {}/{}".format(location, i+1, len(by_location))):
                im = cv2.imread(os.path.join(img_loc, im_id + ".jpg"))
                # TODO append datetime features
                output[im_id] = get_feats_for_im(predictor, im).detach().cpu().numpy()
                flipped_output[im_id] = get_feats_for_im(predictor, cv2.flip(im, 1)).detach().cpu().numpy()
                
            with open(os.path.join(bank_dir, str(location) + ".pkl"), "wb") as f:
                pickle.dump(output, f)
                
            with open(os.path.join(bank_dir, str(location) + "_flip.pkl"), "wb") as f:
                pickle.dump(flipped_output, f)
                
def build_one_bank(cfg, dataset_name, data_dir, bank_dir, loc, in_train):
    """Method used mostly for testing."""
    if not os.path.exists(bank_dir):
        os.mkdir(bank_dir)
        
    predictor = DefaultPredictor(cfg)
    register_dataset(data_dir, dataset_name)
    img_loc = os.path.join(data_dir, _DATASETS[dataset_name]["imgs_loc"])
    
    dataset = dataset_name + "_train" if in_train else dataset_name + "_val"
    dataset_dict = DatasetCatalog.get(dataset)
    df = pd.DataFrame(dataset_dict)
    df = df[df.location == loc]
    if len(df):
        output = {} 
        flipped_output = {}
        im_ids = df.image_id.values
        for im_id in tqdm(im_ids):
            im = cv2.imread(os.path.join(img_loc, im_id + ".jpg"))
            # TODO append datetime features
            output[im_id] = get_feats_for_im(predictor, im).detach().cpu().numpy()
            flipped_output[im_id] = get_feats_for_im(predictor, cv2.flip(im, 1)).detach().cpu().numpy()
                
        with open(os.path.join(bank_dir, str(loc) + ".pkl"), "wb") as f:
            pickle.dump(output, f)

        with open(os.path.join(bank_dir, str(loc) + "_flip.pkl"), "wb") as f:
            pickle.dump(flipped_output, f)
    else:
        print("Location", loc, "not found in", dataset)