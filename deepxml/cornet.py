import torch
import torch.nn as nn
import torch.nn.functional as F

ACT2FN = {"elu": F.elu, "relu": F.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}


class CorNetBlock(nn.Module):
    def __init__(self, context_size, output_size, cornet_act="sigmoid", **kwargs):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = ACT2FN[cornet_act]

    def forward(self, output_dstrbtn):
        identity_logits = output_dstrbtn
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits
        return output_dstrbtn


class CorNet(nn.Module):
    def __init__(self, output_size, cornet_dim=1000, n_cornet_blocks=2, **kwargs):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.ModuleList(
            [
                CorNetBlock(cornet_dim, output_size, **kwargs)
                for _ in range(n_cornet_blocks)
            ]
        )
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)
        return logits


class CorNetWrapper(nn.Module):
    def __init__(self, backbone=None, backbone_fn=None, labels_num=0, **kwargs):
        super(CorNetWrapper, self).__init__()
        if backbone:
            self.backbone = backbone.cuda()
        elif backbone_fn:
            self.backbone = backbone(labels_num, **kwargs)
        else:
            raise Exception
        self.cornet = CorNet(labels_num, **kwargs).cuda()

    def forward(self, input_variables):
        raw_logits = self.backbone(input_variables)
        cor_logits = self.cornet(raw_logits)
        return cor_logits
