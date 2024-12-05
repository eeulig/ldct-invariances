"""Code in this file is heavily based on the code from the following repository: https://github.com/CompVis/invariances"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d


class DenseEncoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, out_size, in_channels=None, width_multiplier=1):
        super().__init__()
        self.scale = scale
        self.wm = width_multiplier
        self.in_channels = int(self.wm * 64 * min(2 ** (self.scale - 1), 16))
        if in_channels is not None:
            print("Warning: Ignoring `scale` parameter in DenseEncoderLayer due to given number of input channels.")
            self.in_channels = in_channels
        self.out_channels = out_size
        self.kernel_size = spatial_size
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            ]
        )


def l2normalize(v, eps=1e-4):
    return v / (v.norm() + eps)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        _w = w.view(height, -1)
        for _ in range(self.power_iterations):
            v = l2normalize(torch.matmul(_w.t(), u))
            u = l2normalize(torch.matmul(_w, v))

        sigma = u.dot((_w).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_dim, activation=F.relu):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.theta = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
        self.phi = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False))
        self.pool = nn.MaxPool2d(2, 2)
        self.g = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1, bias=False))
        self.o_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        N = height * width

        theta = self.theta(x)
        phi = self.phi(x)
        phi = self.pool(phi)
        phi = phi.view(m_batchsize, -1, N // 4)
        theta = theta.view(m_batchsize, -1, N)
        theta = theta.permute(0, 2, 1)
        attention = self.softmax(torch.bmm(theta, phi))  # BX (N) X (N)
        g = self.pool(self.g(x)).view(m_batchsize, -1, N // 4)
        attn_g = torch.bmm(g, attention.permute(0, 2, 1)).view(m_batchsize, -1, width, height)
        out = self.o_conv(attn_g)
        return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = BatchNorm2d(num_features, affine=False, eps=1e-4)
        self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
        self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ConditionalActNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = ActNorm(num_features)
        self.gamma_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))
        self.beta_embed = SpectralNorm(nn.Linear(num_classes, num_features, bias=False))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class BatchNorm2dWrap(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = BatchNorm2d(*args, **kwargs)

    def forward(self, x, y=None):
        return self.bn(x)


class ActNorm2dWrap(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = ActNorm(*args, **kwargs)

    def forward(self, x, y=None):
        return self.bn(x)


class GBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=[3, 3],
        padding=1,
        stride=1,
        n_class=None,
        bn=True,
        activation=F.relu,
        upsample=True,
        downsample=False,
        z_dim=148,
        use_actnorm=False,
        conditional=True,
    ):
        super().__init__()

        self.conv0 = SpectralNorm(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=True if bn or use_actnorm else True)
        )
        self.conv1 = SpectralNorm(
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding, bias=True if bn or use_actnorm else True)
        )

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            if conditional:
                self.HyperBN = ConditionalBatchNorm2d(in_channel, z_dim)
                self.HyperBN_1 = ConditionalBatchNorm2d(out_channel, z_dim)
            else:
                self.HyperBN = BatchNorm2dWrap(in_channel, z_dim)
                self.HyperBN_1 = BatchNorm2dWrap(out_channel, z_dim)
        else:
            if use_actnorm:
                if conditional:
                    self.HyperBN = ConditionalActNorm2d(in_channel, z_dim)
                    self.HyperBN_1 = ConditionalActNorm2d(out_channel, z_dim)
                else:
                    self.HyperBN = ActNorm2dWrap(in_channel)
                    self.HyperBN_1 = ActNorm2dWrap(out_channel)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        # return out
        if self.upsample:
            # different form papers
            out = F.interpolate(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                # different form papers
                skip = F.interpolate(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)
        else:
            skip = input
        return out + skip


class Generator128(nn.Module):
    def __init__(self, code_dim, z_dim, im_ch, n_class, chn, use_actnorm, cond_ch):
        super().__init__()

        G_block_z_dim = code_dim + 28
        self.GBlock = nn.ModuleList(
            [
                GBlock(16 * chn + cond_ch, 16 * chn, n_class=n_class, z_dim=G_block_z_dim),
                GBlock(16 * chn + cond_ch, 8 * chn, n_class=n_class, z_dim=G_block_z_dim),
                GBlock(8 * chn + cond_ch, 4 * chn, n_class=n_class, z_dim=G_block_z_dim),
                GBlock(4 * chn + cond_ch, 2 * chn, n_class=n_class, z_dim=G_block_z_dim),
                GBlock(2 * chn + cond_ch, 1 * chn, n_class=n_class, z_dim=G_block_z_dim),
            ]
        )

        self.sa_id = 4
        self.num_split = len(self.GBlock) + 1
        self.linear = nn.Linear(n_class, 128, bias=False)
        self.first_view = 16 * chn
        first_split = z_dim - (self.num_split - 1) * 20
        self.split_at = [first_split] + [20 for i in range(self.num_split - 1)]
        self.G_linear = SpectralNorm(nn.Linear(first_split, 16 * self.first_view))

        self.attention = SelfAttention(2 * chn)
        if not use_actnorm:
            self.ScaledCrossReplicaBN = BatchNorm2d(1 * chn, eps=1e-4)
        else:
            self.ScaledCrossReplicaBN = ActNorm(1 * chn)
        self.colorize = nn.Conv2d(1 * chn + cond_ch, im_ch, [3, 3], padding=1)

    def forward(self, input, class_id, im_cond=None):
        codes = torch.split(input, self.split_at, 1)
        class_emb = self.linear(class_id)  # 128

        out = self.G_linear(codes[0])

        out = out.view(-1, 4, 4, self.first_view).permute(0, 3, 1, 2)
        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            if im_cond is not None:
                im_cond_rescaled = F.interpolate(im_cond, scale_factor=(1 / 2 ** (len(self.GBlock) - i)))
                out = torch.cat([out, im_cond_rescaled], 1)
            out = GBlock(out, condition)

        out = self.ScaledCrossReplicaBN(out)
        if im_cond is not None:
            out = torch.cat([out, im_cond], 1)
        out = self.colorize(out)
        return out

    def encode(self, *args, **kwargs):
        raise Exception("Sorry, I'm a GAN and not very helpful for encoding.")

    def decode(self, z, cls):
        z = z.float()
        cls_one_hot = torch.nn.functional.one_hot(cls, num_classes=1000).float()
        return self.forward(z, cls_one_hot)
