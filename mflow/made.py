import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask import get_mask, MaskedLinear


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu'):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None):
        h = self.joiner(inputs, cond_inputs)
        m, a = self.trunk(h).chunk(2, 1)
        u = (inputs - m) * torch.exp(-a)
        return u, -a.sum(-1, keepdim=True)

    def inverse(self, inputs, cond_inputs=None):
        x = torch.zeros_like(inputs)
        for i_col in range(inputs.shape[1]):
            h = self.joiner(x, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            x[:, i_col] = inputs[:, i_col] * torch.exp(
                a[:, i_col]) + m[:, i_col]
        return x, -a.sum(-1, keepdim=True)