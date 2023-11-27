import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ACT2FN = {'elu': F.elu, 'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh}

class TreeNetBlock(nn.Module):
    def __init__(self, context_size, input_size, output_size, treenet_act='sigmoid', **kwargs):
        super(TreeNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(input_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = ACT2FN[treenet_act]
    
    def forward(self, output_dstrbtn):        
        identity_logits = output_dstrbtn        
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits        
        return output_dstrbtn
    
    
class TreeNet(nn.Module):
    def __init__(self, input_size, output_size, treenet_dim=1000, n_treenet_blocks=2, **kwargs):
        super(TreeNet, self).__init__()
        layer_sizes = np.linspace(input_size, output_size, 1+n_treenet_blocks)
        self.intlv_layers = nn.ModuleList([TreeNetBlock(treenet_dim, layer_sizes[w], layer_sizes[w+1], **kwargs) for w in range(n_treenet_blocks)])
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):        
        for layer in self.intlv_layers:
            logits = layer(logits)        
        return logits
