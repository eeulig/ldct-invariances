import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from ldctbench.hub import load_model
from torch.autograd import Variable
from tqdm import tqdm

from ldctinv import utils
from ldctinv.data.LDCTMayo import LDCTMayo
from ldctinv.pretrained import load_pretrained
from ldctinv.utils.auxiliaries import apply_windowing

CENTER_WIDTH = (1024 - 600, 1500)  # Center and width of the image (HU values have offset of 1024)


def plot_images(imgs, im_idx, fname, num_invariances=3, figsize=(3, 1)):
    fig, axs = plt.subplots(ncols=4 + num_invariances, figsize=figsize)

    # Plot x
    axs[0].imshow(apply_windowing(imgs["x"][im_idx], *CENTER_WIDTH), cmap="gray", vmin=0, vmax=1.0)
    axs[0].set_title(r"$x$")
    # Plot y
    axs[1].imshow(apply_windowing(imgs["y"][im_idx], *CENTER_WIDTH), cmap="gray", vmin=0, vmax=1.0)
    axs[1].set_title(r"$y$")
    # Plot y_hat
    axs[2].imshow(apply_windowing(imgs["y_hat"][im_idx], *CENTER_WIDTH), cmap="gray", vmin=0, vmax=1.0)
    axs[2].set_title(r"$\hat{y}$")
    # Plot invariances
    for i_inv in range(num_invariances):
        axs[3 + i_inv].imshow(apply_windowing(imgs["inv"][im_idx][i_inv], *CENTER_WIDTH), cmap="gray", vmin=0, vmax=1.0)
        if i_inv == num_invariances // 2:
            axs[3 + i_inv].set_title(r"$\tilde{x}^{(i,k)}$")
    # Plot standard deviation
    std_im = np.std(np.stack(imgs["inv"][im_idx]), 0)
    axs[3 + num_invariances].imshow(std_im, vmin=0, vmax=300)
    axs[3 + num_invariances].set_title(rf"$\sigma ( \{{ \tilde{{x}}^{{(i,k)}} \}}_{{k=1}}^{{{args.n_invariances}}})$")

    for ax in axs:
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


class Invariances(object):
    def __init__(self, run=None, model=None, datafolder=None, seed=1332):
        self.run = run
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not run and not model:
            raise ValueError("Either run or model must be specified")
        elif run and model:
            raise ValueError(f"Only one of run or model can be specified. Got {run} and {model} instead!")
        elif model:
            self.name = model
            # Load models
            nets, data_attr = load_pretrained(model, eval=True, device=self.dev)
            self.greybox, self.vae, self.cinn = nets["greybox"], nets["vae"], nets["cinn"]
            self.vae_domain = "x"  # All models in paper are trained on input domain x
            # Setup dataset
            if not datafolder:
                raise ValueError("If using pretrained models, `datafolder` must be specified")
            data_args = argparse.Namespace(**data_attr)
            data_args.datafolder = datafolder
            data_args.seed = seed
            self.data = LDCTMayo(mode="test", args=data_args)
        elif run:
            self.name = run
            # Setup cINN
            self.cinn, cinn_args = utils.setup_trained_model(
                self.run,
                self.dev,
                network_name="ConditionalTransformer",
                state_dict="cinn",
                return_args=True,
            )
            self.cinn.to(self.dev)
            self.cinn.eval()

            # Setup VAE
            self.vae, vae_args = utils.setup_trained_model(
                cinn_args.vae,
                self.dev,
                network_name="BigAE",
                state_dict="generator",
                in_ch=2,
                out_ch=1,
                cond_ch=1,
                return_args=True,
            )
            self.vae.to(self.dev)
            self.vae.eval()
            self.vae_domain = vae_args.domain

            # Setup greybox model
            self.greybox = load_model(vae_args.greybox, eval=True).to(self.dev)

            # Setup dataset
            if datafolder:
                cinn_args.datafolder = datafolder
            self.data = LDCTMayo(mode="test", args=cinn_args)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @torch.no_grad()
    def get_samples(self, n_samples, n_invariances):
        tags = ["x", "vae_x", "inv", "y", "y_hat", "f_inv"]
        images = {tag: [] for tag in tags}
        images["inv"] = [[] for _ in range(n_samples)]
        images["f_inv"] = [[] for _ in range(n_samples)]
        metrics = {"pix_md": np.zeros(shape=(n_samples, n_invariances), dtype=np.float32)}

        def push_img(**kwargs):
            for tag, img in kwargs.items():
                if tag in ["inv", "f_inv"]:
                    images[tag][img[0]].append(img[1].squeeze().cpu().numpy())
                else:
                    images[tag].append(img.squeeze().cpu().numpy())

        for i in tqdm(range(n_samples), desc=f"Generate invariances"):
            # Get input and target
            batch = self.data[i]
            x, y = batch["x"], batch["y"]
            x = Variable(torch.unsqueeze(x, 0)).to(self.dev)
            y = Variable(torch.unsqueeze(y, 0)).to(self.dev)

            # Apply denoising network
            y_hat = self.greybox(x)

            # Apply VAE
            ae_inp = torch.cat([x if self.vae_domain == "x" else y, y_hat], 1)
            z = self.vae.encode(ae_inp).sample()
            vae_x = self.vae.decode(z, im_cond=y_hat)

            # Apply cINN
            zz, _ = self.cinn(z, y_hat)

            push_img(
                x=self.data.denormalize(x),
                vae_x=self.data.denormalize(vae_x),
                y=self.data.denormalize(y),
                y_hat=self.data.denormalize(y_hat),
            )

            # Sample invariances
            for j in range(n_invariances):
                zz_sample = torch.randn_like(zz)
                zae_sample = self.cinn.reverse(zz_sample, y_hat)
                inv = self.vae.decode(zae_sample, im_cond=y_hat)
                f_inv = self.greybox(inv)
                metrics["pix_md"][i, j] = (x - inv).abs().mean().cpu().numpy().item()
                push_img(
                    inv=(i, self.data.denormalize(inv)),
                    f_inv=(i, self.data.denormalize(f_inv)),
                )

        return images, metrics


def main(args):
    # Reconstruct invariances
    inv = Invariances(run=args.run, model=args.model, datafolder=args.datafolder, seed=args.seed)
    imgs, metrics = inv.get_samples(
        n_samples=args.n_samples,
        n_invariances=args.n_invariances,
    )

    if args.outdir:
        print(f"Saving images and metrics to {args.outdir}")
        # Store images and metrics to files
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        utils.save_obj(imgs, os.path.join(args.outdir, f"{inv.name}-images.pkl"))
        utils.save_obj(metrics, os.path.join(args.outdir, f"{inv.name}-metrics.pkl"))
        # Plot images
        args.n_plots = min(args.n_plots, args.n_samples)
        for im_idx in range(args.n_plots):
            plot_images(
                imgs, im_idx, os.path.join(args.outdir, f"{inv.name}-{im_idx}.png"), num_invariances=3, figsize=(7, 2)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # If using custom trained models
    parser.add_argument(
        "--run",
        default="",
        help="Run name of the trained cINN. Only provide when using custom trained models (see README), default=''",
    )
    # If using pretrained models
    parser.add_argument(
        "--model",
        default="",
        help="Model name of the pretrained model. Only provide when using pretrained models (see README), default=''",
    )
    # Other arguments
    parser.add_argument(
        "--datafolder",
        default="",
        help="Path to datafolder. If provided, will overwrite the datafolder provided during training. If using pretrained models, this must be specified or the LDCT_DATA environment variable must be set, default=''",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional directory to store sampled invariances and images, default=''",
    )
    parser.add_argument(
        "--n_plots",
        type=int,
        default=10,
        help="Number of images to plot (for this --outdir must be set), default=10",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples (patches) for which invariances should be reconstructed, default=10",
    )
    parser.add_argument(
        "--n_invariances",
        type=int,
        default=100,
        help="Number of invariances to sample, default=100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1332,
        help="Random seed to use, default=1332",
    )
    args = parser.parse_args()
    if not hasattr(args, "datafolder") or not args.datafolder:
        if args.run:
            args.datafolder = None
            warnings.warn(f"No datafolder in args. Will use the one provided during training of {args.run}")
        elif args.model and "LDCT_DATA" in os.environ:
            args.datafolder = os.environ["LDCT_DATA"]
            warnings.warn(
                f"No datafolder in args. Will use the one provided via environment variable LDCT_DATA: {args.datafolder}"
            )
        else:
            raise ValueError(
                "No datafolder provided! Add via\n\t- Arguments: add argument --datafolder\n\t- Environment variable: export LDCT_DATA=..."
            )
    main(args)
