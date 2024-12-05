import argparse
import importlib
import os
import warnings
from argparse import Namespace
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .auxiliaries import load_yaml


def setup_trained_model(
    run_name,
    device=torch.device("cuda"),
    network_name="Model",
    state_dict=None,
    return_args=False,
    return_model=True,
    pretrained_run=None,
    **model_kwargs,
) -> Union[Tuple[nn.Module, argparse.Namespace], nn.Module, argparse.Namespace]:
    savepath = os.path.join("wandb", "run-" + run_name, "files")
    if not os.path.exists(savepath):
        warnings.warn(f"Folder run-{run_name} not found in wandb directory. We'll try finding an offline-run!")
        savepath = os.path.join("wandb", "offline-run-" + run_name, "files")
        if not os.path.exists(savepath):
            raise FileNotFoundError(f"Neither found  run-{run_name} nor offline-run-{run_name} in wandb directory!")
    args = argparse.Namespace(**load_yaml(os.path.join(savepath, "args.yaml")))
    if return_args and not return_model:
        return args
    model_class = getattr(importlib.import_module("ldctinv.{}.network".format(args.trainer)), network_name)
    model_args = args
    if pretrained_run:
        model_args = argparse.Namespace(
            **load_yaml(os.path.join("wandb", "run-" + pretrained_run, "files", "args.yaml"))
        )
    model = model_class(model_args, **model_kwargs).to(device)
    if state_dict:
        state = torch.load(os.path.join(savepath, "{}.pt".format(state_dict)), weights_only=True)
        print(f"Restore state dict {state_dict} of {network_name} from epoch", state["epoch"])
        model.load_state_dict(state["model_state_dict"])
    if return_args:
        return model, args
    return model


def setup_dataloader(args: Namespace, datasets: Dict[str, Dataset]) -> Dict[str, DataLoader]:
    """Returns dict of dataloaders

    Parameters
    ----------
    args : Namespace
        Command line arguments
    datasets : Dict[str, Dataset]
        Dictionary of datasets for each phase.

    Returns
    -------
    Dict[str, DataLoader]
        Dictionray of dataloaders for each phase.
    """
    dataloader = {}
    for phase, data in datasets.items():
        sampler = WeightedRandomSampler(data.weights, len(data), replacement=True)
        dataloader[phase] = DataLoader(
            dataset=data,
            batch_size=args.mbs,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
            drop_last=True,
            sampler=sampler,
        )
    return dataloader


def setup_optimizer(args, parameters):
    """Setup the optimizer to use"""

    if args.optimizer == "sgd":
        return optim.SGD(parameters, lr=args.lr, momentum=args.sgd_momentum, dampening=args.sgd_dampening)
    elif args.optimizer == "adam":
        return optim.Adam(parameters, lr=args.lr, betas=(args.adam_b1, args.adam_b2))
    elif args.optimizer == "rmsprop":
        return optim.RMSprop(parameters, lr=args.lr)
    else:
        raise ValueError("Optimizer unknown. Must be one of sgd | adam | rmsprop")


def l1(input, target):
    return torch.abs(input - target)


def l2(input, target):
    return torch.pow((input - target), 2)


def setup_loss(loss):
    """Setup the loss to use"""
    try:
        return globals()[loss]
    except:
        raise ValueError("Loss function unknown.")


class AELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pixel_loss = setup_loss(args.criterion)

    def forward(self, reconstructions, targets, posteriors, is_kl=True):
        rec_loss = self.pixel_loss(reconstructions, targets)
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]
        if is_kl:
            kl_loss = posteriors
        else:
            kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        return rec_loss, kl_loss


def nll(sample):
    return 0.5 * torch.sum(torch.pow(sample, 2), dim=[1, 2, 3])


class UnsupervisedINNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample, logdet):
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        return loss


class normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        for ch in range(3):
            x[:, ch, :, :] = (x[:, ch, :, :] - self.mean[ch]) / self.std[ch]
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"


class repeat_ch(object):
    def __init__(self, in_ch):
        self.in_ch = in_ch

    def __call__(self, x):
        if self.in_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"


class PerceptualLoss(torch.nn.Module):
    def __init__(self, network, device, in_ch=3, layers=[3, 8, 15, 22], norm="l1", return_features=False):
        super(PerceptualLoss, self).__init__()
        """Network can be either vgg16 or vgg19. The layer defines where to
        extract the activations. In the default paper, style losses are computed
        at:
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3'
            '22': "relu4_3"
        and perceptual (content) loss is evaluated at: '15': "relu3_3". In the
        Yang et al., 2018 paper (https://arxiv.org/pdf/1708.00961.pdf) content
        is evaluated in vgg19 after the 16th (last) conv layer (layer '35')"""

        if network == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        elif network == "vgg19":
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
        vgg.eval()

        self.vgg_features = vgg.features
        self.layers = [str(l) for l in layers]
        if norm == "l1":
            self.norm = nn.L1Loss()
        elif norm == "mse":
            self.norm = nn.MSELoss()
        else:
            raise ValueError("Norm {} not known for PerceptualLoss".format(norm))
        self.transform = repeat_ch(in_ch)
        self.return_features = return_features

    def forward(self, input, target):
        input = self.transform(input)
        target = self.transform(target)

        loss = 0.0
        if self.return_features:
            features = {"input": [], "target": []}

        for i, m in self.vgg_features._modules.items():
            input = m(input)
            target = m(target)

            if i in self.layers:
                loss += self.norm(input, target)
                if self.return_features:
                    features["input"].append(input.clone())
                    features["target"].append(target.clone())

                if i == self.layers[-1]:
                    break

        return (loss, features) if self.return_features else loss
