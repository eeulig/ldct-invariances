# Reconstructing and Analyzing the Invariances of LDCT Denoising Networks
[![DOI](https://img.shields.io/badge/10.1002%2Fmp.17413-red?label=Paper)](https://doi.org/10.1002/mp.17413)

This repository contains code for the paper:
> Elias Eulig, Fabian Jäger, Joscha Maier, Björn Ommer, and Marc Kachelrieß. Reconstructing and analyzing the invariances of low-dose CT image denoising networks. Medical Physics. 2024.

## Table of Contents
- [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Download the data](#download-the-data)
  * [(Optional) Set environment variable to the data folder](#optional-set-environment-variable-to-the-data-folder)
- [Reconstruct invariances of LDCT denoising networks](#reconstruct-invariances-of-ldct-denoising-networks)
  * [Using pretrained models](#using-pretrained-models)
  * [Using custom trained models](#using-custom-trained-models)
- [Reference](#reference)

## Getting Started
### Installation
1. Clone this repository: `git clone https://github.com/eeulig/ldct-invariances`
2. Go into the new folder: `cd ldct-invariances`
3. Install the correct PyTorch version for your operating system and CUDA version from [PyTorch](https://pytorch.org/get-started/locally/) directly.
4. Install the package with `pip install .`

If you want to make changes to the codebase you should install the package in editable mode using
```sh
pip install -e .
```
and consider installing the optional development dependencies by running
```sh
pip install .[dev]
```

### Download the data
In our paper we reconstruct the invariances of LDCT denoising networks on the chest scans of the Low Dose CT and Projection data[^1] which is available from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/). For downloading the data, please follow the instructions below.

**Sign a TCIA license agreement:** You must sign and submit a TCIA Restricted License Agreement to download the data. Information on how to do this is provided under "Data Access" [here](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/).

**Download the LDCT data:** We provide the `.tcia` manifest file containing the series IDs of the Siemens chest scans under `assets/manifest.tcia`. You can download the data to some folder `/path/to/datafolder` using the `ldctbench-download-data` command line tool (part of the [ldct-benchmark](https://www.github.com/eeulig/ldct-benchmark) package) by running
```sh
ldctbench-download-data --savedir /path/to/datafolder --username <username> --password <password>
```
You must provide your TCIA username and password to access the data.
> [!NOTE]
> If your username or password contains special characters, you may need to enclose them in single quotes.

Alternatively, you can use the [NBIA data retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Version+4.4) and provide it the `assets/manifest.tcia` file to download the data.

### (Optional) Set environment variable to the data folder
Running training and reconstruction code is easiest with the data folder set as an environment variable. You can set the environment variable by running
```sh
export LDCT_DATA=/path/to/datafolder
```
where `/path/to/datafolder` is the path to the downloaded data folder. Alternatively, you can provide the path to the data folder in the config `.yaml` file for training models or via the argument `--datafolder` when calling `train.py` or `reconstruct.py`.

## Reconstruct invariances of LDCT denoising networks
### Using pretrained models
We provide weights for the conditional VAE and the cINN to reconstruct invariances of the following algorithms:
| Weights Name | Denoising Network | Paper                                                                                                                                                                                                                                                                                              |
|--------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| cnn10        | CNN-10            | H. Chen, Y. Zhang, W. Zhang, P. Liao, K. Li, J. Zhou, and G. Wang, "Low-dose CT via convolutional neural network,” Biomedical Optics Express, vol. 8, no. 2, pp. 679–694, Jan. 2017                                                                                                                |
| redcnn       | RED-CNN           | H. Chen, Y. Zhang, M. K. Kalra, F. Lin, Y. Chen, P. Liao, J. Zhou, and G. Wang, “Low-dose CT with a residual encoder-decoder convolutional neural network,” IEEE Transactions on Medical Imaging, vol. 36, no. 12, pp. 2524–2535, Dec. 2017                                                        |
| wganvgg      | WGAN-VGG          | Q. Yang, P. Yan, Y. Zhang, H. Yu, Y. Shi, X. Mou, M. K. Kalra, Y. Zhang, L. Sun, and G. Wang, “Low-dose CT image denoising using a generative adversarial network with wasserstein distance and perceptual loss,” IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1348– 1357, Jun. 2018. |
| dugan        | DU-GAN            | Z. Huang, J. Zhang, Y. Zhang, and H. Shan, “DU-GAN: Generative adversarial networks with dual-domain U-Net-based discriminators for low-dose CT denoising,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–12, 2022.                                                         |

To reconstruct the invariances of, e.g. CNN-10, you can load the associated models using
```python
from ldctinv.pretrained import load_pretrained
nets, data_attr = load_pretrained("cnn10")
greybox, vae, cinn = nets["greybox"], nets["vae"], nets["cinn"]
```
We provide code to reconstruct the invariances using these pretrained models in `reconstruct.py`. Simply run
```sh
python reconstruct.py --model cnn10 --outdir results/cnn10
```
to reconstruct the invariances of the CNN-10 model. This will create 12 files in a folder `results/cnn10`.
- `cnn10-0.png` - `cnn10-9.png`: Plots of the reconstructed invariances
- `cnn10-images.pkl`: Pickled file containing the results. Let `n_samples`, `n_invariances` be the number of sampled patches and number of sampled invariances, respectively. Then, the file contains a dictionary with following keys:
  - `x`: Low-dose input images (list of `n_samples` images)
  - `vae_x`: Low-dose input images reconstructed by the VAE (list of `n_samples` images)
  - `y`: High-dose target images (list of `n_samples` images)
  - `y_hat`: Low-dose input images denoised by the denoising network `f` (list of `n_samples` images)
  - `inv`: Reconstructed invariances (list of lists of `n_samples` x `n_invariances` images)
  - `f_inv`: Reconstructed invariances fed back into the denoising network (list of lists of `n_samples` x `n_invariances` images)
- `cnn10-metrics.pkl`: Pickled file containing a dictionary with the following keys:
  - `pix_md`: Pixel-wise mean absolute difference between sampled invariances and low-dose input images (numpy array of shape `n_samples` x `n_invariances`)

### Using custom trained models
You can train your own VAE and cINN models using the `train.py` script. The following sections assume that you set the environment variable for the data folder as described [here](#optional-set-environment-variable-to-the-data-folder). For example, to retrain the models for the CNN-10 algorithm, you can run
#### Train cVAE
```sh
python train.py --configs/cnn10-vae.yaml
```
#### Train cINN
```sh
echo -e '\nvae: <run-name-vae>' >> configs/cnn10-cinn.yaml
python train.py --configs/cnn10-cinn.yaml
```
where `<run-name-vae>` is the name of the cVAE training run.

#### Reconstruct invariances
Then reconstruct the invariances using the trained models by running
```sh
python reconstruct.py --run <run-name-cinn> --outdir results/cnn10-retrained
```
with `<run-name-cinn>` being the name of the cINN training.

## Acknowledgements
The code for training the cVAE and cINN is based the code from Rombach et al., 2020[^3] available [here](https://github.com/CompVis/invariances).

## Reference
If you find this project useful for your work, please cite our [Medical Physics paper](https://doi.org/10.1002/mp.17413):

```bibtex
@article{ldctinv-medphys,
    author = {Eulig, Elias and Jäger, Fabian and Maier, Joscha and Ommer, Björn and Kachelrieß, Marc},
    title = {Reconstructing and analyzing the invariances of low-dose CT image denoising networks},
    journal = {Medical Physics},
    year = {2024},
    doi = {https://doi.org/10.1002/mp.17413},
    url = {https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.17413},
}
```

[^1]: McCollough, C., Chen, B., Holmes III, D. R., Duan, X., Yu, Z., Yu, L., Leng, S., & Fletcher, J. (2020). Low Dose CT Image and Projection Data (LDCT-and-Projection-data) (Version 6) [Data set]. The Cancer Imaging Archive. <https://doi.org/10.7937/9NPB-2637>.

[^3]: Rombach, Robin, Patrick Esser, and Bjorn Ommer. 2020. "Making Sense of CNNs: Interpreting Deep Representations & Their Invariances with INNs." In European Conference on Computer Vision (ECCV), 18.
