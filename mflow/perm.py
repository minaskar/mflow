import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Perm(nn.Module):
    def __init__(self, nvars, perm=None):
        super(self.__class__, self).__init__()
        # If perm is none, chose some random permutation that gets fixed at initialization
        if perm is None:
            perm = torch.randperm(nvars)
        self.perm = perm
        self.reverse_perm = torch.argsort(perm)

    def forward(self, x, context):
        idx = self.perm.to(x.device)
        return x[:, idx], 0

    def inverse(self, x, context):
        rev_idx = self.reverse_perm.to(x.device)
        return x[:, rev_idx], 0