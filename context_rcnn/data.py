import torch
import pandas as pd
import random
import numpy as np
import copy
import os
import pickle
import sys
import io
import logging
from typing import List, Optional, Union
from datetime import datetime

import contextlib
from detectron2.config import configurable
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper, utils, T
from detectron2.structures import BoxMode
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
from fvcore.transforms.transform import HFlipTransform


_DATASETS = {
    # relative to data_dir
    "cct": {
        "imgs_loc": "cct/train_val/",
        "labels_loc": "cct/",
        
        # 1 class
        "default": {
            "train_filename": "cct_coco_cameratraps_animal_train.json",
            "val_filename": "cct_coco_cameratraps_animal_val.json",
            "num_classes": 1
        },
        
        # all classes
        "species": {
            "train_filename": "cct_coco_cameratraps_species_train.json",
            "val_filename": "cct_coco_cameratraps_species_val.json",
            "num_classes": 21
        },
        
        # for testing - contains 1924 images
        "toy1924": {
            "train_filename": "cct_coco_cameratraps_animal_toy1924.json",
            "val_filename": "cct_coco_cameratraps_animal_toy1924.json",
            "num_classes": 1
        },
        
        # for testing - contains 1924 images
        "toy1924-species": {
            "train_filename": "cct_coco_cameratraps_species_toy1924.json",
            "val_filename": "cct_coco_cameratraps_species_toy1924.json",
            "num_classes": 21
        },
        
        # for testing - one month of toy1924
        "toy-month": {
            "train_filename": "cct_coco_cameratraps_species_toy_month.json",
            "val_filename": "cct_coco_cameratraps_species_toy_month.json",
            "num_classes": 21
        },
    }
}
logger = logging.getLogger(__name__)

def register_dataset(data_dir, dataset_name, subset="default"):
    """Register a dataset in _DATASETS, or re-register it if it already exists.
    
    Return:
        (train_key, val_key, num_classes)
    """
    dataset = _DATASETS[dataset_name]
    imgs_path = os.path.join(data_dir, dataset["imgs_loc"])
    labels_path = os.path.join(data_dir, dataset["labels_loc"])

    train_coco_path = os.path.join(labels_path, dataset[subset]["train_filename"])
    val_coco_path = os.path.join(labels_path, dataset[subset]["val_filename"])
    num_classes = dataset[subset]["num_classes"]

    train_key = dataset_name + "_train"
    val_key = dataset_name + "_val"

    try:
        DatasetCatalog.remove(train_key)
        MetadataCatalog.remove(train_key)
    except KeyError:
        pass
    register_coco_cameratraps_instances(train_key, {}, train_coco_path, imgs_path)    
    
    try:
        DatasetCatalog.remove(val_key)
        MetadataCatalog.remove(val_key)
    except KeyError:
        pass
    register_coco_cameratraps_instances(val_key, {}, val_coco_path, imgs_path)

    return (train_key, val_key, num_classes)

def register_coco_cameratraps_instances(name, metadata, json_file, image_root):
    """
    See detectron2.data.datasets.register.coco -> register_coco_instances(...)
    
    Modified to call load_coco_cameratraps_json instead of load_coco_json.
    
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_cameratraps_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )
    
def load_coco_cameratraps_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    See detectron2.data.datasets.coco.load_coco_json(...)
    
    Minor modifications here to include datetime and location, since Detectron2 methods don't allow
    for extra annotation keys at the image level.
    
    Comments and spacing removed for brevity. Location of modification is marked.
    """
    from pycocotools.coco import COCO
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))
    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    if "minival" not in json_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )
    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    if "subcategories" in coco_api.dataset.keys():
        ann_keys.append("subcategory_id")
    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        
        ### MODIFICATION HERE - optional coco_cameratraps info ###
        record["location"] = img_dict.get("location", None)
        record["datetime"] = img_dict.get("date_captured", None)
        ### END MODIFICATION ###

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0
            obj = {key: anno[key] for key in ann_keys if key in anno}
            segm = anno.get("segmentation", None)
            if segm:
                if not isinstance(segm, dict):
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm
            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts

class ContextDatasetMapper(DatasetMapper):
    """
    A DatasetMapper which also loads relevant feature banks, based on image location
    and whether it has been augmented with horizontal flipping.
    
    Otherwise, uses default logic from DatasetMapper.
    """
    @configurable
    def __init__(
        self,
        *,
        banks_dir: str,
        num_context_items: int,
        num_context_feats: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.banks_dir = banks_dir
        self.num_context_items = num_context_items
        self.num_context_feats = num_context_feats
        
    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = super().from_config(cfg, is_train)
        ret.update({
            "banks_dir": cfg.MODEL.CONTEXT.BANKS_DIR,
            "num_context_items": cfg.MODEL.CONTEXT.NUM_CONTEXT_ITEMS,
            "num_context_feats": cfg.MODEL.CONTEXT.NUM_CONTEXT_FEATS
        })
        return ret
       
    def _get_bank_feats(self, location, month, flipped):
        """Load memory bank features for this location from disk."""
        loc_filename = os.path.join(self.banks_dir, location + ("_flip.pkl" if flipped else ".pkl"))
        loc_feats_dict = {}
        try:
            with open(loc_filename, "rb") as loc_file:
                loc_feats_dict = pickle.load(loc_file)
        except:
            print("Bank error loc", str(loc_filename), sys.exc_info()[0])
        
        # populate feature tensors
        final_feats = torch.zeros([1, self.num_context_items, self.num_context_feats], 
                                 dtype=torch.float)
        
        num_valid_context_items = 0
        if len(loc_feats_dict.values()):
            month_feats = loc_feats_dict[month]
            if len(month_feats):
                loc_feats = torch.stack([torch.as_tensor(v) for v in month_feats.values()])
                num_valid_context_items = min(self.num_context_items, len(loc_feats))
                # TODO if more than self.num_context_feats, randomly select?
                loc_feats = loc_feats[:num_valid_context_items]
                final_feats[0, :len(loc_feats), :] = loc_feats
            else:
                # remove this
                print("No feats for month", month, "at loc", location)
        else:
            print("No feats for loc", loc_filename)
        
        return final_feats, torch.tensor([num_valid_context_items], dtype=int)
    
    def __call__(self, dataset_dict):
        """
        Small modifications from DatasetMapper.__call__
        Comments and spacing removed for brevity. Location of modification is marked.
        
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        ### begin modifications for ContextDatasetMapper ###
        has_flip = any(isinstance(t, HFlipTransform) for t in transforms)
        month = datetime.fromisoformat(dataset_dict["datetime"]).month
        context_feats, num_valid_context_items = self._get_bank_feats(dataset_dict["location"], month, has_flip)
        dataset_dict["context_feats"] = context_feats
        dataset_dict["num_valid_context_items"] = num_valid_context_items
        dataset_dict["banks_dir"] = self.banks_dir
        dataset_dict["has_flip"] = has_flip
        ### end modifications for ContextDatasetMapper ###
        
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
        

class LocationSampler(torch.utils.data.sampler.Sampler):
    """
    Sample randomly by location to ensure each batch is, when possible,
    from the same location, to limit i/o in loading location-based
    feature banks.
    
    TODO: not currently used
    """
    def __init__(self, df, batch_size):
        """
        Args:
            df: a DataFrame with a column 'location' and indices
                which match the indices that should be returned by self.__iter__
        """
        self.df = df
        self.batch_size = batch_size

    def __iter__(self):
        locs = set(self.df.location.unique().tolist())
        inds = set(self.df.index.values)
        final_order = []

        while len(final_order) < len(self.df) and len(locs) > 0:
            loc = random.choice(list(locs))
            valid_inds = [ ind for ind in self.df.index[self.df.location == loc].tolist() if ind in inds ]
            if (len(valid_inds) <= self.batch_size):
                locs.remove(loc)
            valid_inds = valid_inds[:self.batch_size]
            for i in valid_inds:
                inds.remove(i)
                final_order.append(self.df.index.get_loc(i))
            
        return iter(final_order)

    def __len__(self):
        return len(self.df)