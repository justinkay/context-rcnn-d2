from typing import Dict, List, Union, Optional, Tuple
import torch
import math
import numpy as np

from detectron2.config import configurable
import detectron2.modeling.proposal_generator.proposal_utils as proposal_utils
from detectron2.structures import Instances, ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from .attention import Attention


@ROI_HEADS_REGISTRY.register()
class ContextROIHeads(StandardROIHeads):
    
    @configurable
    def __init__(
        self,
        *,
        long_term: Attention,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.long_term = long_term
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            "long_term": Attention.from_config(cfg)
        })
        return ret
    
    def forward(
        self,
        batched_inputs,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        The same as StandardRoiHeads.forward, but passes batched_inputs to _forward_box.
        TODO find a way around this.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(batched_inputs, features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(batched_inputs, features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
        
        
    def _forward_box(self, batched_inputs, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Adds Context R-CNN logic to StandardROIHeads._forward_box. Modifications are indicated.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        
        ## begin Context R-CNN logic ##
        
        # pooling concatenates all batches - split again by image
        batch_size = len(proposals)
        num_boxes, num_filters, kernel1, kernel2 = box_features.shape
        box_features = torch.reshape(box_features, [batch_size, -1, num_filters, kernel1, kernel2])
        context_features = torch.cat([i["context_feats"] for i in batched_inputs])
        num_valid_context_items = torch.cat([i["num_valid_context_items"] for i in batched_inputs])
        
        # "Finally, we add F context as a per-feature-channel bias back into our original input features A"
        box_features += self.long_term(box_features, context_features, num_valid_context_items)
        
        # re-combine all images of batch
        box_features = torch.reshape(box_features, [num_boxes, num_filters, kernel1, kernel2])
        
        del context_features
        del num_valid_context_items
        
        ## end Context R-CNN logic ##
        
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances