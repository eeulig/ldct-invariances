import os
import random
from argparse import Namespace
from typing import Dict, List

# import nibabel as nib
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

from ldctinv.utils import load_yaml


class LDCTMayo(Dataset):
    """Dataset class for the LDCT dataset"""

    def __init__(self, mode: str, args: Namespace):
        """Init function

        Parameters
        ----------
        mode : str
            Which subset to use. Must be `train`, `val`, or `test`.
        args : Namespace
            Command line argumetns passed to the dataset class.

        Attributes
        ----------
        seed: int
            Random seed to use
        path: str
            Root path to datafolder
        patchsize: int
            Patchsize to use for training.
        data_subset: float
            Subset of the data to use.
        data_norm: str
            Normalization of the data, must be `meanstd` or `minmax`.
        info: dict
            Dictionary of `info.yml` associated with the dataset.
        samples: list
            List of all image slices
        weights: list
            List of weights associated with each slice. Slices of each patient having `n_slices` are weighted with `1/n_slices`.

        Examples
        --------
        Create a training dataset

        >>> from argparse import Namespace
        >>> from ldctbench.data import LDCTMayo
        >>> args = Namespace(seed=1332, path='/path/to/dataset/root', data_norm='meanstd', data_subset=1.0, patchsize=128)
        >>> train_data = LDCTMayo('train', args)
        """

        # Set seeds
        self.seed = args.seed
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.path = args.datafolder
        self.patchsize = args.patchsize
        self.data_subset = args.data_subset
        self.data_norm = args.data_norm
        self.info = load_yaml(os.path.join(os.path.dirname(os.path.realpath(__file__)), "info.yml"))

        # Get all slices == samples for this split
        self.samples = [
            {**patient_dict, "slice": s + 1}
            for patient_dict in self.info[mode + "_set"]
            for s in range(patient_dict["n_slices"])
        ]
        random.shuffle(self.samples)

        self.weights = torch.tensor(
            [1.0 / patient_dict["n_slices"] for patient_dict in self.samples],
            dtype=torch.double,
        )

        if self.data_subset < 1.0:
            self.samples = self.samples[: int(len(self.samples) * self.data_subset)]
            self.weights = self.weights[: int(len(self.weights) * self.data_subset)]

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize samples with precomputed mean/std or min/max

        Parameters
        ----------
        X : np.ndarray
            Array to normalize

        Returns
        -------
        np.ndarray
            Normalized array

        Raises
        ------
        ValueError
            If normalization method `self.data_norm` is neither meanstd nor minmax
        """
        if self.data_norm == "meanstd":
            return (X - self.info["mean"]) / self.info["std"]
        elif self.data_norm == "minmax":
            return (X - float(self.info["min"])) / (float(self.info["max"]) - float(self.info["min"]))
        else:
            raise ValueError(f"Unknown normalization method {self.data_norm}")

    def denormalize(self, X: np.ndarray) -> np.ndarray:
        """Denormalize samples with precomputed mean/std or min/max

        Parameters
        ----------
        X : np.ndarray
            Array to denormalize

        Returns
        -------
        np.ndarray
            Denormalized array

        Raises
        ------
        ValueError
            If normalization method `self.data_norm` is neihter meanstd nor minmax
        """
        if self.data_norm == "meanstd":
            return X * self.info["std"] + self.info["mean"]

        elif self.data_norm == "minmax":
            return X * (self.info["max"] - self.info["min"]) + self.info["min"]
        else:
            raise ValueError(f"Unknown normalization method {self.data_norm}")

    @staticmethod
    def _idx2filename(idx: int, n_slices: int) -> str:
        """Get filename for a patient of LDCT data given slice and number of slices in the scan.

        Parameters
        ----------
        idx : int
            Slice idx for which to return filename
        n_slices : int
            Number of slices of the scan necessary to figure out number of trailing zeros

        Returns
        -------
        str
            Filename
        """
        return "1-{}.dcm".format(str(idx).zfill(len(str(n_slices))))

    def _random_crop(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Randomly crop the same patch from images (e.g. ground truth and input)

        Parameters
        ----------
        images : List[np.ndarray]
            List of images to crop in same position for

        Returns
        -------
        List[np.ndarray]
            List of cropped images
        """
        assert images[0].shape == images[1].shape, "Both images must have same shape!"

        if (self.patchsize != images[0].shape[0] or self.patchsize != images[0].shape[1]) and self.patchsize:
            x = np.random.randint(images[0].shape[0] - self.patchsize)
            y = np.random.randint(images[0].shape[1] - self.patchsize)
            images = [im[x : x + self.patchsize, y : y + self.patchsize] for im in images]
        return images

    @staticmethod
    def to_torch(X: np.ndarray) -> torch.Tensor:
        """Convert input image to torch tensor and unsqueeze"""
        return torch.unsqueeze(torch.from_numpy(X), 0)

    def reset_seed(self):
        """Reset random seeds"""
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __len__(self):
        """Length of dataset"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Given an index, get slice, perform random cropping, normalize and return"""

        # Load image
        sample = self.samples[idx]
        f_name = self._idx2filename(sample["slice"], sample["n_slices"])
        x = pydicom.filereader.dcmread(os.path.join(self.path, sample["input"][2:], f_name)).pixel_array.astype(
            "float32"
        )
        y = pydicom.filereader.dcmread(os.path.join(self.path, sample["target"][2:], f_name)).pixel_array.astype(
            "float32"
        )

        # Crop gt and input
        x, y = self._random_crop([x, y])

        return {
            "x": self.to_torch(self._normalize(x)),
            "y": self.to_torch(self._normalize(y)),
        }
