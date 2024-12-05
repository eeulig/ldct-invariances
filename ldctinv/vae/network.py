"""Code in this file is heavily based on the code from the following repository: https://github.com/CompVis/invariances"""

import warnings

import torch
import torch.nn as nn
from torchvision import models

from ldctinv.utils.distributions import DiagonalGaussianDistribution
from ldctinv.vae.blocks import ActNorm, DenseEncoderLayer, Generator128


class ClassUp(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_sigmoid=False, out_dim=None):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x.squeeze(-1).squeeze(-1))
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class BigGANDecoderWrapper(nn.Module):
    """Wraps a BigGAN into our autoencoding framework"""

    def __init__(self, im_ch, chn, cond_ch, z_dim, input_size, decoder_norm):
        super().__init__()
        self.z_dim = z_dim
        image_size = input_size
        assert image_size == 128, "We only support 128x128 images / patches"
        use_actnorm = True if decoder_norm == "an" else False
        class_embedding_dim = 1000
        self.extra_z_dims = list()

        self.map_to_class_embedding = ClassUp(
            z_dim, depth=2, hidden_dim=2 * class_embedding_dim, use_sigmoid=False, out_dim=class_embedding_dim
        )
        self.decoder = Generator128(
            120, z_dim, im_ch, n_class=class_embedding_dim, chn=chn, use_actnorm=use_actnorm, cond_ch=cond_ch
        )

    def forward(self, x, labels=None, im_cond=None):
        emb = self.map_to_class_embedding(x[:, : self.z_dim, ...])
        x = self.decoder(x, emb, im_cond=im_cond)
        return x


_norm_options = {"in": nn.InstanceNorm2d, "bn": nn.BatchNorm2d, "an": ActNorm}


class ResnetEncoder(nn.Module):
    def __init__(self, in_ch, z_dim, input_size, norm, net_type="resnet50", pretrained=False):
        super().__init__()
        __possible_resnets = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
        }
        __possible_weights = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
        }
        z_dim = self.z_dim = z_dim
        ipt_size = input_size
        self.input_size = [in_ch, ipt_size, ipt_size]
        norm_layer = _norm_options[norm]
        self.z_dim = z_dim
        if pretrained and norm != "bn":
            warnings.warn(
                f"If pretrained, then args.norm must be bn! Got pretrained={pretrained} but norm_layer={norm}. Using bn for ResNet instead!"
            )
            norm_layer = _norm_options["bn"]
        self.model = __possible_resnets[net_type](
            weights=__possible_weights[net_type] if pretrained else None, norm_layer=norm_layer
        )

        # replace first conv
        self.model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        size_pre_fc = self._get_spatial_size()
        assert size_pre_fc[2] == size_pre_fc[3], "Output spatial size is not quadratic"
        spatial_size = size_pre_fc[2]
        num_channels_pre_fc = size_pre_fc[1]

        # replace last fc
        self.model.fc = DenseEncoderLayer(
            0, spatial_size=spatial_size, out_size=2 * z_dim, in_channels=num_channels_pre_fc
        )

    def forward(self, x):
        features = self.features(x)
        encoding = self.model.fc(features)
        return encoding

    def features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x

    def post_features(self, x):
        x = self.model.fc(x)
        return x

    def _get_spatial_size(self):
        x = torch.randn(1, *self.input_size)
        return self.features(x).size()


class BigAE(nn.Module):
    def __init__(self, args, in_ch, cond_ch, **kwargs):
        super().__init__()
        self.cond_ch = cond_ch
        input_size = args.patchsize

        out_ch = kwargs["out_ch"] if "out_ch" in kwargs else in_ch

        self.encoder = ResnetEncoder(
            in_ch, args.z_dim, input_size, args.encoder_norm, args.encoder_type, args.encoder_pretrained
        )
        self.decoder = BigGANDecoderWrapper(
            out_ch, args.decoder_chn, cond_ch, args.z_dim, input_size, args.decoder_norm
        )

    def encode(self, input):
        h = input
        h = self.encoder(h)
        return DiagonalGaussianDistribution(h, deterministic=False)

    def decode(self, input, im_cond=None):
        h = input
        h = self.decoder(h.squeeze(-1).squeeze(-1), im_cond=im_cond)
        return h

    def forward(self, inputs, return_z=False, im_cond=None):
        assert (im_cond is None) == (not self.cond_ch)
        if im_cond is not None:
            inputs = torch.cat([inputs, im_cond], 1)

        posterior = self.encode(inputs)
        z = posterior.sample()
        reconstructions = self.decode(z, im_cond=im_cond)
        if return_z:
            return posterior.kl(), reconstructions, z
        return posterior.kl(), reconstructions
