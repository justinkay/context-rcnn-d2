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

from .fbo import SimpleFBO


@ROI_HEADS_REGISTRY.register()
class ContextROIHeads(StandardROIHeads):
    
    @configurable
    def __init__(
        self,
        *,
        fbo: SimpleFBO,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fbo = fbo
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            "fbo": SimpleFBO(cfg)
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
        This moves the logic for when box features are pooled to before the
        sampling of proposals for training; this leads to increased computations
        for pooling but is essential to keep the contextual memory banks a constant 
        size for training and evaluation.
        """
        # pool proposal box features
        _features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(_features, [x.proposal_boxes for x in proposals])
        
        # the Context R-CNN step
        box_features = self.fbo(batched_inputs, box_features)
        
        del images
        
        if self.training:
            assert targets
            num_proposals_per_img = len(proposals[0])
            num_gt_per_img = [len(x.gt_boxes) for x in targets]
            
            if self.proposal_append_gt:
                # pool ground truth box features
                gt_features = self.box_pooler(_features, [x.gt_boxes for x in targets])
                gt_features_per_img = torch.split(gt_features, num_gt_per_img)

                # interleave box_features and gt_features, so that this will match up
                # with the list of proposals when gt boxes are added and sampled
                box_features_per_img = torch.split(box_features, num_proposals_per_img)
                all_box_features = []
                for i in range(len(box_features_per_img)):
                    all_box_features.append(box_features_per_img[i])
                    if len(gt_features_per_img[i]):
                        all_box_features.append(gt_features_per_img[i])
                box_features = torch.cat(all_box_features)
            
            # sample proposals to have similar foreground and background examples.
            # gt boxes are added if self.proposal_append_gt==True
            proposals, sampled_idxs = self.label_and_sample_proposals(proposals, targets)
            # idxs are per-image; add correction for stacking the idx tensors
            last_ind = 0
            for i, img_idxs in enumerate(sampled_idxs):
                sampled_idxs[i] = img_idxs + last_ind
                last_ind = last_ind + num_proposals_per_img
                if self.proposal_append_gt:
                    last_ind = last_ind + num_gt_per_img[i]
            
            # sample box_features
            sampled_idxs = torch.cat(sampled_idxs)
            box_features = box_features[sampled_idxs]
            
        del targets

        if self.training:
            losses = self._forward_box(box_features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(box_features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
        
    def _forward_box(
        self, box_features, proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        See StandardROIHeads._forward_box
        
        This essentially has the same logic, but in order to avoid pooling twice, it
        takes already-pooled box features as a parameter. The labeling and sampling
        of proposals is expected to have already happened.
        """
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
        
    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]):
        """
        Same as ROIHeads.label_and_sample_proposals, except also returns sampled
        indices.
        
        Comments and spacing removed for brevity.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        all_sampled_idxs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            all_sampled_idxs.append(sampled_idxs)
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        
        return proposals_with_gt, all_sampled_idxs
    
    def disable(self):
        """Test method; permanently disables the FBO."""
        print("Disabling the FBO")
        from context_rcnn.fbo import DisabledFBO
        self.fbo = DisabledFBO()
        
        
