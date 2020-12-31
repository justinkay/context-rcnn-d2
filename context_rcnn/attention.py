import torch
import torch.nn.functional as F


class FullyConnected(torch.nn.Module):
    """
    A fully connected layer (linear, relu, bn) plus additional
    l2 normalization as in the tf implementation.
    """
    def __init__(self, D_in, D_out, normalize=False):
        super(FullyConnected, self).__init__()
        
        self.D_out = D_out
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_out),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(D_out)
        )
        self.normalize = normalize
        
    def forward(self, x):
        batch_size, _, num_features = x.shape
        features = torch.reshape(x, [-1, num_features])
        
        y = self.fc(features)
        y = torch.reshape(y, [batch_size, -1, self.D_out])
        
        if self.normalize:
            y = F.normalize(y, dim=-1, p=2)
            
        return y

class Attention(torch.nn.Module):
    def __init__(self, n, m, inp_d0, con_d0, d1=2048, d2=2048, temp=0.01, normalize_vf=True):
        """
        Args:
            n: number of input items (bounding boxes)
            m: number of context items (Mshort or Mlong in paper)
            inp_d0: number of features per input box (2048 in paper (original Faster RCNN implementation);
                                                      1024 for Detectron2 FRCNN C4; 
                                                      256 for FPN)
            con_d0: number of features per context item (d0 in paper)
            d1: first hidden layer size   (2048 in paper)
            d2: second hidden layer size  (2048 in paper)
            temp: softmax temperature     (0.01 in paper)
            normalize_vf: whether to l2 normalize the outputs of values and final projection layers.
                            This appears to be True in the tf implementation, but False in the paper's
                            diagram, and not specified in paper's implementation details
        """
        super(Attention, self).__init__()
        
        self.fc_q = FullyConnected(inp_d0, d1, normalize=True)
        self.fc_k = FullyConnected(con_d0, d1, normalize=True)
        self.fc_v = FullyConnected(con_d0, d2, normalize=True)
        self.fc_f = FullyConnected(d2, inp_d0, normalize=False)
        
        self.temp = temp

    def forward(self, x, x_context, num_valid_context_items):
        """
        Args:
            x: B x n x inp_d0 x kernel_size x kernel_size
            x_context: B x m x con_d0
            num_valid_context_items: B x 1; how many (<= m) context items are present
        
        Return:
            input + f_context: B x n x inp_d0 x kernel_size x kernel_size
        """
        A_pool = x.mean([-2, -1])                             # -> B x n x inp_d0
        
        queries = self.fc_q(A_pool)                           # -> B x n x d1
        keys = self.fc_k(x_context)                           # -> B x m x d1
        values = self.fc_v(x_context)                         # -> B x m x d2
        weights = torch.bmm(queries, keys.transpose(-2,-1))   # -> B x n x m
        
        # mask attention weights and values. probably slow; TODO do batch in one step
        for im in range(x.shape[0]):
            weights[im, :, num_valid_context_items[im]:] = float('-inf')
            values[im, num_valid_context_items[im]:, :] = 0
        
        weights = F.softmax(weights / self.temp, dim=-1)      # -> B x n x m, feats for each n sum to 1
        wv = torch.bmm(weights, values)                       # -> B x n x d2
        f_context = self.fc_f(wv)                             # -> B x n x inp_d0
        
        # store these for debugging and visualization
        self._last_weights = weights
        
        # "Finally, we add F context as a per-feature-channel bias back into our original input features A"
        f_context_bias = f_context.unsqueeze(-1).unsqueeze(-1).expand(x.shape) # -> B x n x inp_d0 x kernel_size x kernel_size
        
        return x + f_context_bias
        