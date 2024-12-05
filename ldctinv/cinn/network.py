"""Code in this file is heavily based on the code from the following repository: https://github.com/CompVis/invariances"""

import numpy as np
import torch
import torch.nn as nn

from ldctinv.cinn.blocks import ConditionalFlow, FeatureLayer
from ldctinv.vae.blocks import ActNorm, DenseEncoderLayer


class DenseEmbedder(nn.Module):
    """Basically an MLP. Maps vector-like features to some other vector of given dimenionality"""

    def __init__(self, in_dim, up_dim, depth=4, given_dims=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == in_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(in_dim, up_dim, depth).astype(int)
        for l in range(len(dims) - 2):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            self.net.append(ActNorm(dims[l + 1]))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Conv2d(dims[-2], dims[-1], 1))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x.squeeze(-1).squeeze(-1)


class Embedder(nn.Module):
    """Embeds a 4-dim tensor onto dense latent code."""

    def __init__(self, in_spatial_size, in_channels, emb_dim, n_down=4):
        super().__init__()
        self.feature_layers = nn.ModuleList()
        norm = "an"  # hard coded
        bottleneck_size = in_spatial_size // 2**n_down
        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm))
        self.dense_encode = DenseEncoderLayer(n_down, bottleneck_size, emb_dim)
        if n_down == 1:
            print(
                " Warning: Embedder for ConditionalTransformer has only one down-sampling step. You might want to "
                "increase its capacity."
            )

    def forward(self, input):
        h = input
        for layer in self.feature_layers:
            h = layer(h)
        h = self.dense_encode(h)
        return h.squeeze(-1).squeeze(-1)


class ConditionalTransformer(nn.Module):
    def __init__(self, args):
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True
        super().__init__()
        conditioning_option = "none"
        flowactivation = "lrelu"
        embedding_channels = args.in_channels

        self.flow = ConditionalFlow(
            in_channels=args.in_channels,
            embedding_dim=embedding_channels,
            hidden_dim=args.mid_channels,
            hidden_depth=args.hidden_depth,
            n_flows=args.n_flows,
            conditioning_option=conditioning_option,
            activation=flowactivation,
        )
        self.embedder = Embedder(
            args.conditioning_spatial_size, args.conditioning_in_ch, args.in_channels, n_down=args.n_down
        )

    def embed(self, conditioning):
        # embed it via embedding layer
        embedding = self.embedder(conditioning)
        return embedding

    def sample(self, shape, conditioning):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, self.embed(conditioning))
        return sample

    def forward(self, input, conditioning, train=False):
        embedding = self.embed(conditioning)
        out, logdet = self.flow(input, embedding)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, conditioning):
        embedding = self.embed(conditioning)
        return self.flow(out, embedding, reverse=True)

    def get_last_layer(self):
        return getattr(self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight")
