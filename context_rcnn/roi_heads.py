from typing import Dict, List, Union, Optional, Tuple
import torch
import math
import numpy as np

from detectron2.config import configurable
import detectron2.modeling.proposal_generator.proposal_utils as proposal_utils
from detectron2.structures import Instances, ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads, Res5ROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from .attention import Attention


def add_context(pooled_features, context_features, num_valid_context_items, batch_size, attention):
    """
    n = num input proposals
    m = num context proposals
    Args:
        pooled_features: torch.Tensor, B*n x num_filters x kernel_size x kernel size. Pooled backbone features of RPN
                        proposals, output from an ROIPooler.
        context_features: torch.Tensor, B x m x num_context_feats
        num_valid_context_items: torch.Tensor, B x 1; how many (<= m) context items are present for each image
        batch_size: int
        attention: Attention
    """
    # pooling concatenates all batches - split again by image
    num_boxes, num_filters, kernel_size1, kernel_size2 = pooled_features.shape
    pooled_features = torch.reshape(pooled_features, [batch_size, -1, num_filters, kernel_size1, kernel_size2])
        
    # pool over spatial dimensions
    pooled_features = pooled_features.mean([-2,-1])
        
    # "Finally, we add F context as a per-feature-channel bias back into our original input features A"
    pooled_features = pooled_features + attention(pooled_features, context_features, num_valid_context_items)
        
    # re-combine all images of batch
    pooled_features = torch.reshape(pooled_features, [-1, num_filters])
        
    # and expand to match original size
    pooled_features = pooled_features.unsqueeze(-1).unsqueeze(-1).expand([num_boxes, num_filters, kernel_size1, kernel_size2])
    
    return pooled_features


@ROI_HEADS_REGISTRY.register()
class ContextROIHeads(StandardROIHeads):
    """
    The ROI heads class used with FPN backbones.
    """
    
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
        
        This adds context features after pooling backbone features but before the box_head.
        """
        features = [features[f] for f in self.box_in_features]
        
        ## begin Context R-CNN logic ##
        
        batch_size = len(proposals)
        context_features = torch.cat([i["context_feats"] for i in batched_inputs])
        num_valid_context_items = torch.cat([i["num_valid_context_items"] for i in batched_inputs])
        
        pooled_features = self.box_pooler(features, [x.proposal_boxes for x in proposals]) # -> B*n x 256 x 7 x 7
        pooled_features = add_context(pooled_features, context_features, num_valid_context_items, batch_size, self.long_term)
        box_features = self.box_head(pooled_features)
        
        del pooled_features
        del context_features
        del num_valid_context_items
        
        ## end Context R-CNN logic ##
        
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
        

@ROI_HEADS_REGISTRY.register()
class ContextRes5ROIHeads(Res5ROIHeads):
    """
    The ROIHeads class which emulates the original Faster R-CNN paper. It passes backbone features
    through a res5 block before the box_predictor.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.long_term = Attention.from_config(cfg)
    
    def forward(self, batched_inputs, images, features, proposals, targets=None):
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        ## begin Context R-CNN logic - replaces self._shared_roi_transform
        
        batch_size = len(proposals)
        context_features = torch.cat([i["context_feats"] for i in batched_inputs])
        num_valid_context_items = torch.cat([i["num_valid_context_items"] for i in batched_inputs])
        
        features = [features[f] for f in self.in_features]
        pooled_features = self.pooler(features, [x.proposal_boxes for x in proposals]) # -> B*n x 1024 x 7 x 7
        
        pooled_features = add_context(pooled_features, context_features, num_valid_context_items, batch_size, self.long_term)
        box_features = self.res5(pooled_features)
        
        del pooled_features
        del context_features
        del num_valid_context_items
    
        ## end Context R-CNN logic
    
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}