import os
import pickle
from typing import Tuple, Union

import numpy as np
import yaml


def load_yaml(path):
    with open(path) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
    return content


def dump_config(args, path):
    with open(os.path.join(path, "args.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_windowing(
    x: np.ndarray,
    center: Union[int, float, None] = None,
    width: Union[int, float, None] = None,
    out_range: Tuple = (0, 1),
) -> np.ndarray:
    """Apply some windowing (center and width) to an image

    Parameters
    ----------
    x : np.ndarray
        Image array of arbitrary dimension
    center : int, float, optional
        Float indicating the center
    width : int, float, optional
        Float indicating the width
    out_range : Tuple, optional
        Desired output range, by default (0, 1)

    Returns
    -------
    np.ndarray
        Copy of input array with center and width applied
    """
    # Check if arguments are valid
    if center is None and width is None:
        raise ValueError("Center and width must be provided.")

    center = float(center)
    width = float(width)

    lower = center - 0.5 - (width - 1) / 2.0
    upper = center - 0.5 + (width - 1) / 2.0
    res = np.empty(x.shape, dtype=x.dtype)
    res[x <= lower] = out_range[0]
    res[(x > lower) & (x <= upper)] = ((x[(x > lower) & (x <= upper)] - (center - 0.5)) / (width - 1.0) + 0.5) * (
        out_range[1] - out_range[0]
    ) + out_range[0]
    res[x > upper] = out_range[1]
    return res
