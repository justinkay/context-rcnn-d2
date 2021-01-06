import torch
import torch.nn.functional as F


class FullyConnected(torch.nn.Module):
    """
    A fully connected layer (linear, relu, bn) plus additional l2 normalization.
    """
    def __init__(self, D_in, D_out, normalize=False):
        super(FullyConnected, self).__init__()
        
        self.D_in = D_in
        self.D_out = D_out
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_out),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(D_out, eps=0.001, momentum=0.97) # match tf settings
        )
        self.normalize = normalize
        
    def forward(self, x):
        """
        Args:
            x: B x n x D_in
        Return:
            y: B x n x D_out
        """
        batch_size, _, num_features = x.shape
        assert(num_features == self.D_in), str(num_features) + " does not match required input size to FC layer:" + "str(self.D_in)"
        
        features = torch.reshape(x, [-1, num_features])
        
        y = self.fc(features)
        y = torch.reshape(y, [batch_size, -1, self.D_out])
        
        if self.normalize:
            y = F.normalize(y, dim=-1, p=2)
            
        return y

    
class Attention(torch.nn.Module):
    def __init__(self, num_input_feats, num_context_feats, d1=2048, d2=2048, temp=0.01):
        """
        Args:
            num_feats: number of features per input box (2048 in paper/original Faster RCNN implementation)
            num_context_feats: number of features per context item (d0=2057 in paper)
            d1: first hidden layer size   (2048 in paper)
            d2: second hidden layer size  (2048 in paper)
            temp: softmax temperature     (0.01 in paper)
        """
        super(Attention, self).__init__()
        
        self.fc_q = FullyConnected(num_input_feats, d1, normalize=True)
        self.fc_k = FullyConnected(num_context_feats, d1, normalize=True)
        self.fc_v = FullyConnected(num_context_feats, d2, normalize=True)
        self.fc_f = FullyConnected(d2, num_input_feats, normalize=False)
        
        self.temp = temp
        
    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.MODEL.CONTEXT.NUM_INPUT_FEATS, cfg.MODEL.CONTEXT.NUM_CONTEXT_FEATS,
                       cfg.MODEL.CONTEXT.D1, cfg.MODEL.CONTEXT.D2, cfg.MODEL.CONTEXT.SOFTMAX_TEMP)

    def forward(self, x, x_context, num_valid_context_items):
        """
        n = num input proposals
        m = num context proposals
        Args:
            x: B x n x num_input_feats
            x_context: B x m x num_context_feats
            num_valid_context_items: B x 1; how many (<= m) context items are present for each image
        
        Return:
            f_context: B x n x num_input_feats
        """
        queries = self.fc_q(x)                                # -> B x n x d1
        keys = self.fc_k(x_context)                           # -> B x m x d1
        values = self.fc_v(x_context)                         # -> B x m x d2
        weights = torch.bmm(queries, keys.transpose(-2,-1))   # -> B x n x m
        
        # mask attention weights and values
        weights_mask = torch.ones_like(weights)
        values_mask = torch.ones_like(values)
        for im in range(x.shape[0]):
            weights_mask[im, :, num_valid_context_items[im]:] = 0
            values_mask[im, num_valid_context_items[im]:, :] = 0
        weights = weights.masked_fill(weights_mask == 0, -1e9)
        values = values.masked_fill(values_mask == 0, 0)

        weights = F.softmax(weights / self.temp, dim=-1)      # -> B x n x m, feats for each n sum to 1
        wv = torch.bmm(weights, values)                       # -> B x n x d2
        f_context = self.fc_f(wv)                             # -> B x n x num_input_feats
        
        # store these for debugging and visualization
        self._last_weights = weights.detach()
        self._last_bias = f_context.detach()
        
        return f_context
        