import argparse
import hashlib
import importlib
import os
from typing import Optional

import requests
import torch
from ldctbench.hub import load_model

torch.serialization.add_safe_globals([argparse.Namespace])
import torch.nn as nn
import tqdm
from ldctbench.utils import load_json
from platformdirs import user_cache_dir

CHECKPOINTS = load_json(os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints.json"))


def download_checkpoint(name: str, url: str, checksum: str) -> str:
    cache_dir = user_cache_dir(appname="ldctinv", appauthor="eeulig")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    checkpoint_name = name + ".pt"
    checkpoint_path = os.path.join(cache_dir, checkpoint_name)

    if os.path.exists(checkpoint_path):
        print(f"Found {checkpoint_name} in {cache_dir}!")
        return checkpoint_path

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("content-length", 0))
    file = open(os.path.join(checkpoint_path), "wb")
    sha256 = hashlib.sha256()
    with tqdm.tqdm(
        desc=f"Download {checkpoint_name} to {cache_dir}",
        total=file_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            size = file.write(chunk)
            sha256.update(chunk)
            bar.update(size)
    digest = sha256.hexdigest()
    file.close()
    if digest != checksum:
        raise RuntimeError(f'invalid hash value (expected "{checksum}", got "{digest}")')
    return checkpoint_path


def load_pretrained(
    weights_name: str,
    eval: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a pretrained model

    Parameters
    ----------
    weights_name : str
        String, specifying the name of the weights to load
    eval : bool
        Return network in eval mode, by default True
    device: torch.device, optional

    Returns
    -------
    nn.Module
        The pretrained model

    """

    if not device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cfg = CHECKPOINTS[weights_name]

    # Get denoising network from ldct-benchmark
    models = dict(greybox=load_model(weights_name, eval=eval, device=device))

    data_attr = cfg.pop("data_attr", None)
    for net in cfg:
        model_class = getattr(
            importlib.import_module(f"ldctinv.{net}.network"),
            cfg[net]["model_name"],
        )
        model = model_class(argparse.Namespace(**cfg[net]["args"]), **cfg[net]["kwargs"]).to(device)

        # Download checkpoint
        chkpt_path = download_checkpoint(f"{weights_name}-{net}", cfg[net]["url"], checksum=cfg[net]["checksum"])
        state = torch.load(chkpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        if eval:
            model.eval()
        models[net] = model
    return models, data_attr
