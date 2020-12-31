from typing import Dict, List, Union, Optional, Tuple
import torch
import math
import numpy as np

from detectron2.data.detection_utils import convert_image_to_rgb
import detectron2.modeling.proposal_generator.proposal_utils as proposal_utils
from detectron2.structures import Instances, ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class ContextRCNN(GeneralizedRCNN):
    
    def forward(self, batched_inputs):
        """
        Modified to match input parameters to ContextROIHeads, which takes
        in batched_inputs as well.
        
        TODO: probably can get rid of batched_inputs as input to ContextROIHeads if
        moving context_features to self.device here.
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        for i in batched_inputs:
            i["context_feats"] = i["context_feats"].to(self.device)
            
        _, detector_losses = self.roi_heads(batched_inputs, images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Modified to match input parameters to ContextROIHeads, which takes
        in batched_inputs as well.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            for i in batched_inputs:
                i["context_feats"] = i["context_feats"].to(self.device)
            results, _ = self.roi_heads(batched_inputs, images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
        
    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image, up to 20 top-scoring predicted
        object proposals on the original image, and contextual images corresponding to
        the top 4 attention weights. 

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        import cv2
        import os
        import pickle
        from detectron2.utils.visualizer import Visualizer, VisImage

        storage = get_event_storage()
        max_vis_prop = 1

        for input, prop, last_weights in zip(batched_inputs, proposals, self.roi_heads.fbo.long_term._last_weights):
            # from d2; gt and top proposal(s)
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            imw = img.shape[1]
            imh = img.shape[0]
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            # visualize attn weights
            bank = pickle.load(open(os.path.join(input["banks_dir"], str(input["location"]) + ".pkl"), "rb"))
            all_weights = last_weights.flatten()
            top_values, top_indices = torch.topk(all_weights, k=4)
            
            rows = top_indices // last_weights.shape[-1]       # top-weighted proposals
            inds_in_row = top_indices % last_weights.shape[-1] # top-weighted context vectors - only care about these
            print("In RCNN visualize, top attn weights", top_values, "proposal idxs", rows, "context img idxs", inds_in_row)
        
            att_imgs = []
            for ind in inds_in_row:
                top_im_path = os.path.join(os.path.dirname(input["file_name"]), list(bank.keys())[ind] + ".jpg")
                top_im = convert_image_to_rgb(cv2.imread(top_im_path), "BGR")
                resize_to = (imw, imh)
                top_im = cv2.resize(top_im, resize_to)
                
                context_feats = input["context_feats"].squeeze()
                feats_top_im = context_feats[ind].detach().clone()
                cx, cy, w, h = feats_top_im[-4:]
                cx *= imw
                cy *= imh
                w *= imw
                h *= imh
                xmin = int(cx - w/2)
                ymin = int(cy - h/2)
                xmax = int(cx + w/2)
                ymax = int(cy + h/2)
                top_im = cv2.rectangle(top_im, (xmin, ymin), (xmax, ymax),color=(0, 255, 0), thickness=3)
                att_img = VisImage(top_im).get_image()
                att_imgs.append(att_img)

            vis_img = np.concatenate((anno_img, prop_img, *att_imgs), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "From left: GT bounding boxes; Predicted proposals; Top attention weights"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch