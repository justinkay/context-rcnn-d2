import torch

from .attention import Attention


class SimpleFBO(torch.nn.Module):
    def __init__(self, cfg):
        """
        A Feature Bank Operator with no datetime features. Simply grabs as many entries from
        the memory bank as possible.
        """
        super(SimpleFBO, self).__init__()
        self.long_term = Attention(n = cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, # number of proposals
                                    m = cfg.MODEL.CONTEXT.NUM_CONTEXT_ITEMS,
                                    inp_d0 = cfg.MODEL.CONTEXT.NUM_INPUT_FEATS,
                                    con_d0 = cfg.MODEL.CONTEXT.NUM_CONTEXT_FEATS,
                                    d1 = cfg.MODEL.CONTEXT.D1,
                                    d2 = cfg.MODEL.CONTEXT.D2,
                                    temp = cfg.MODEL.CONTEXT.SOFTMAX_TEMP)
        
    def forward(self, batched_inputs, box_features):
        """
        Return a context-biased version of box_features.
        """
        # pooling concatenates all batches - split again by image
        batch_size = len(batched_inputs)
        num_boxes, num_filters, kernel1, kernel2 = box_features.shape
        box_features = torch.reshape(box_features, [batch_size, -1, num_filters, kernel1, kernel2])
        
        context_features = torch.cat([i["context_feats"] for i in batched_inputs])
        num_valid_context_items = torch.cat([i["num_valid_context_items"] for i in batched_inputs])
        
        # get biased features from attention networks and pass to ROI heads
        box_features = self.long_term(box_features, context_features, num_valid_context_items)
        del context_features
        
        # reshape box_features back to how it was
        box_features = torch.reshape(box_features, [num_boxes, num_filters, kernel1, kernel2])
        
        return box_features
    

class DisabledFBO(torch.nn.Module):
    """For testing; similar to torch.nn.Identity but takes 2 input arguments to forward()"""
    def __init__(self, *args, **kwargs):
        super(DisabledFBO, self).__init__()

    def forward(self, batched_inputs, box_features):
        return box_features