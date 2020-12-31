import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
import detectron2.evaluation.coco_evaluation as d2
from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table

from .data import ContextDatasetMapper

    
class COCOEvaluatorWithAR(d2.COCOEvaluator):
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Adds AR stats from the COCO API to COCOEvaluator.
        
        Other than the metrics dict, this method is unchanged from COCOEvaluator.
        Removing comments and spacing for brevity.
        
        AR_n: Average Recall with n boxes
        """
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl",
                    "AR_1", "AR_10", "AR_100", "ARs" "ARm", "ARl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]
        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + d2.create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")
        if class_names is None or len(class_names) <= 1:
            return results
        precisions = coco_eval.eval["precision"]
        assert len(class_names) == precisions.shape[2]
        results_per_category = []
        for idx, name in enumerate(class_names):
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)
        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results
