import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

_norm_options = {"in": nn.InstanceNorm2d, "bn": nn.BatchNorm2d, "sn": spectral_norm}


class Critic(nn.Module):
    def __init__(self, args, in_ch, input_ft, max_ft=265, depth=4, **kwargs):
        super(Critic, self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        in_ft = [in_ch] + [min([max_ft, input_ft * 2**i]) for i in range(depth)]

        f_k_s_list = [
            (in_ft[i], in_ft[i + 1], k, s) for i, k, s in zip(range(len(in_ft) - 1), [4] * depth, [2] * depth)
        ] + [(in_ft[-1], in_ft[-1], 3, 1)]

        def add_block(ch_in, ch_out, kernel, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, 1))
            if "norm" in kwargs:
                assert kwargs["norm"] in _norm_options, "Norm must be in, bn or sn"
                if kwargs["norm"] == "bn" or kwargs["norm"] == "in":
                    layers.append(_norm_options[kwargs["norm"]](ch_out))
                else:
                    layers[-1] = _norm_options[kwargs["norm"]](layers[-1])
            layers.append(self.activation)
            return layers

        layers = []
        for ch_in, ch_out, k, s in f_k_s_list:
            add_block(ch_in, ch_out, k, s)
        layers.append(nn.Conv2d(in_ft[-1], 1, 1, 1, 0))
        self.features = nn.ModuleList(layers)

        w, h = self._comp_output_size(in_ch, args.patchsize)

        self.linear = nn.Linear(w * h, 1)

    def _comp_output_size(self, in_ch, w):
        x = torch.randn(1, in_ch, w, w)
        for i in range(len(self.features)):
            x = self.features[i](x)
        return x.shape[-2:]

    def forward(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
