"""Code in this file is heavily based on the code from the following repository: https://github.com/CompVis/invariances"""

import functools

import torch
import torch.nn as nn

from ldctinv.vae.blocks import ActNorm


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class EfficientVRNVP(nn.Module):
    def __init__(self, module_list, in_channels, n_flow, hidden_dim, hidden_depth):
        super().__init__()
        assert in_channels % 2 == 0
        self.flow = nn.ModuleList([Flow(module_list, in_channels, hidden_dim, hidden_depth) for n in range(n_flow)])

    def forward(self, x, condition=None, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.flow)):
                x, logdet_ = self.flow[i](x, condition=condition)
                logdet = logdet + logdet_
            return x, logdet
        else:
            for i in reversed(range(len(self.flow))):
                x = self.flow[i](x, condition=condition, reverse=True)
            return x, None

    def reverse(self, x, condition=None):
        return self.flow(x, condition=condition, reverse=True)


class ConditionalFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""

    def __init__(
        self,
        in_channels,
        embedding_dim,
        hidden_dim,
        hidden_depth,
        n_flows,
        conditioning_option="none",
        activation="lrelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = embedding_dim
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.conditioning_option = conditioning_option

        self.sub_layers = nn.ModuleList()
        if self.conditioning_option.lower() != "none":
            self.conditioning_layers = nn.ModuleList()
        for flow in range(self.n_flows):
            self.sub_layers.append(
                ConditionalFlatDoubleCouplingFlowBlock(
                    self.in_channels, self.cond_channels, self.mid_channels, self.num_blocks, activation=activation
                )
            )
            if self.conditioning_option.lower() != "none":
                self.conditioning_layers.append(nn.Conv2d(self.cond_channels, self.cond_channels, 1))

    def forward(self, x, embedding, reverse=False):
        hconds = list()
        hcond = embedding[:, :, None, None]
        self.last_outs = []
        self.last_logdets = []
        for i in range(self.n_flows):
            if self.conditioning_option.lower() == "parallel":
                hcond = self.conditioning_layers[i](embedding)
            elif self.conditioning_option.lower() == "sequential":
                hcond = self.conditioning_layers[i](hcond)
            hconds.append(hcond)
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x, hconds[i])
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, hconds[i], reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)


class Flow(nn.Module):
    def __init__(self, module_list, in_channels, hidden_dim, hidden_depth):
        super(Flow, self).__init__()
        self.in_channels = in_channels
        self.flow = nn.ModuleList(
            [module(in_channels, hidden_dim=hidden_dim, depth=hidden_depth) for module in module_list]
        )

    def forward(self, x, condition=None, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.flow)):
                x, logdet_ = self.flow[i](x)
                logdet = logdet + logdet_
            return x, logdet
        else:
            for i in reversed(range(len(self.flow))):
                x = self.flow[i](x, reverse=True)
            return x


class DoubleVectorCouplingBlock(nn.Module):
    """In contrast to VectorCouplingBlock, this module assures alternating chunking in upper and lower half."""

    def __init__(self, in_channels, hidden_dim, depth=2, use_hidden_bn=False, n_blocks=2):
        super(DoubleVectorCouplingBlock, self).__init__()
        assert in_channels % 2 == 0
        self.s = nn.ModuleList(
            [
                BasicFullyConnectedNet(dim=in_channels // 2, depth=depth, hidden_dim=hidden_dim, use_tanh=True)
                for _ in range(n_blocks)
            ]
        )
        self.t = nn.ModuleList(
            [
                BasicFullyConnectedNet(dim=in_channels // 2, depth=depth, hidden_dim=hidden_dim, use_tanh=False)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x, logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x


class ConditionalDoubleVectorCouplingBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, depth=2):
        super(ConditionalDoubleVectorCouplingBlock, self).__init__()
        self.s = nn.ModuleList(
            [
                BasicFullyConnectedNet(
                    dim=in_channels // 2 + cond_channels,
                    depth=depth,
                    hidden_dim=hidden_dim,
                    use_tanh=True,
                    out_dim=in_channels // 2,
                )
                for _ in range(2)
            ]
        )
        self.t = nn.ModuleList(
            [
                BasicFullyConnectedNet(
                    dim=in_channels // 2 + cond_channels,
                    depth=depth,
                    hidden_dim=hidden_dim,
                    use_tanh=False,
                    out_dim=in_channels // 2,
                )
                for _ in range(2)
            ]
        )

    def forward(self, x, xc, reverse=False):
        assert len(x.shape) == 4
        assert len(xc.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        xc = xc.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                scale = self.s[i](conditioner_input)
                x_ = x[idx_keep] * scale.exp() + self.t[i](conditioner_input)
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale, dim=1)
                logdet = logdet + logdet_
            return x[:, :, None, None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                x_ = (x[idx_keep] - self.t[i](conditioner_input)) * self.s[i](conditioner_input).neg().exp()
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:, :, None, None]


class ConditionalFlatDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth, activation="lrelu"):
        super().__init__()
        __possible_activations = {"lrelu": InvLeakyRelu, "none": IgnoreLeakyRelu}
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = ConditionalDoubleVectorCouplingBlock(in_channels, cond_channels, hidden_dim, hidden_depth)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, xcond, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h, xcond)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, xcond, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out, xcond):
        return self.forward(out, xcond, reverse=True)


class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer("forward_shuffle_idx", nn.Parameter(idx, requires_grad=False))
        self.register_buffer("backward_shuffle_idx", nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]


class VectorActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True, **kwargs):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False, conditioning=None):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        if not reverse:
            _, _, height, width = input.shape
            if self.initialized.item() == 0:
                self.initialize(input)
                self.initialized.fill_(1)
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            if not self.logdet:
                return (self.scale * (input + self.loc)).squeeze(3).squeeze(2)
            return (self.scale * (input + self.loc)).squeeze(3).squeeze(2), logdet
        else:
            return self.reverse(input)

    def reverse(self, output, conditioning=None):
        return (output / self.scale - self.loc).squeeze(3).squeeze(2)


class IgnoreLeakyRelu(nn.Module):
    """performs identity op."""

    def __init__(self, alpha=0.9):
        super().__init__()

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        h = input
        return h, 0.0

    def reverse(self, input):
        h = input
        return h


class InvLeakyRelu(nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        scaling = (input >= 0).to(input) + (input < 0).to(input) * self.alpha
        h = input * scaling
        return h, 0.0

    def reverse(self, input):
        scaling = (input >= 0).to(input) + (input < 0).to(input) * self.alpha
        h = input / scaling
        return h


class InvParametricRelu(InvLeakyRelu):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)


class FeatureLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm="AN", width_multiplier=1):
        super().__init__()

        norm_options = {"in": nn.InstanceNorm2d, "bn": nn.BatchNorm2d, "an": ActNorm}

        self.scale = scale
        self.norm = norm_options[norm.lower()]
        self.wm = width_multiplier
        if in_channels is None:
            self.in_channels = int(self.wm * 64 * min(2 ** (self.scale - 1), 16))
        else:
            self.in_channels = in_channels
        self.out_channels = int(self.wm * 64 * min(2**self.scale, 16))
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        self.sub_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                Norm(num_features=self.out_channels),
                Activate(),
            ]
        )
