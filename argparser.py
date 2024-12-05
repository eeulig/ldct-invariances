import argparse

import yaml


def add_vae_arguments(parser):
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Train a VAE conditioned on the greybox predictions (for this --greybox must be set), default=False",
    )
    parser.add_argument("--z_dim", type=int, help="Size of latent space")
    parser.add_argument("--domain", help="Domain in which to train VAE. Must be one of x (low-dose) | y (high-dose)")
    parser.add_argument("--criterion", help="Criterion to use for reconstruction loss. Must be one of l1 | l2")

    # Encoder
    parser.add_argument(
        "--encoder_type",
        help="Encoder to use for VAE training. Must be one of resnet18 | resnet34 | resnet50 | resnet101",
    )
    parser.add_argument(
        "--encoder_norm", help="Normalization to use in encoder. If --encoder_pretrained is set, this must be bn"
    )
    parser.add_argument(
        "--encoder_pretrained",
        dest="encoder_pretrained",
        action="store_true",
        help="Whether to use encoder pretrained on ImageNet",
    )
    # Decoder
    parser.add_argument("--decoder_chn", type=int, help="Number of base channels of the decoder")
    parser.add_argument(
        "--decoder_norm", help="Normalization to use in decoder. Must be one of bn (BatchNorm) | an (ActNorm)"
    )
    # Critic
    parser.add_argument(
        "--cond_critic",
        action="store_true",
        help="Whether to use a conditional critic (conditioned on greybox prediction) for adversarial loss (for this --conditional and --greybox must be set), default=False",
    )
    # Loss
    parser.add_argument("--lam", type=float, help="Gradient penalty coefficient for discriminator")
    parser.add_argument("--kl_weight", type=float, help="Weight of KL-divergence for VAE")
    parser.add_argument("--alpha", type=float, help="Weight of (pixelwise) reconstruction loss for VAE")
    parser.add_argument("--beta", type=float, help="Weight of perceptual loss for VAE")
    return parser


def add_cinn_arguments(parser):
    # VAE
    parser.add_argument("--vae", help="Run name of the VAE")

    # Network
    parser.add_argument(
        "--in_channels",
        type=int,
        help="Embedding dim of the conditional input. Also defines the number of input channels of each fully-connected network (s_i, t_i in Romach et al.) in each coupling block as: in_channels // 2",
    )
    parser.add_argument("--mid_channels", type=int, help="Hidden dim of s_i, t_i.")
    parser.add_argument(
        "--hidden_depth", type=int, help="Number of layers in each fully-connected network in the coupling blocks"
    )
    parser.add_argument("--n_flows", type=int, help="Number of invertible blocks to use")
    parser.add_argument(
        "--n_down",
        type=int,
        help="Downsampling steps of the embedding network (H in Rombach et al.) for the conditioning",
    )
    parser.add_argument(
        "--conditioning_spatial_size", type=int, help="Spatial size of conditioning, 128 for all our experiments"
    )
    parser.add_argument(
        "--conditioning_in_ch", type=int, help="Number of channels of conditioning, 1 in all our experiments"
    )
    return parser


def make_parser():
    parser = argparse.ArgumentParser()

    # -----------------------------   General Setup   ------------------------------#
    parser.add_argument("--config", default="", help="Provide a yaml config file")
    parser.add_argument("--devices", nargs="+", type=int, default=0, help="List of Cuda device to use, default=0")
    parser.add_argument("--trainer", help="Trainer to use. Must be one of vae | cinn")
    parser.add_argument("--loginterv", type=int, default=4, help="How many images to log per epoch (default=4)")
    parser.add_argument("--dryrun", action="store_true", help="Deactivate syncing to Weights & Biases")
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="Use cuda, default=True")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="Deactivate cuda")
    parser.set_defaults(cuda=True)

    # ---------------------------------   Data   ------------------------------------#
    parser.add_argument(
        "--datafolder",
        default="",
        help="Path to datafolder downloaded from TCIA",
    )
    parser.add_argument(
        "--patchsize",
        default=128,
        type=int,
        help="Patchsize used for training. Set to 128 for all our experiments",
    )
    parser.add_argument(
        "--data_subset",
        type=float,
        default=1.0,
        help="Subset of the training data to use (good for debugging), default=1.0",
    )
    parser.add_argument(
        "--data_norm",
        default="meanstd",
        help="Input normalization, must be one of minmax | meanstd, default='meanstd'",
    )

    # ---------------------------   General Training   ------------------------------#
    parser.add_argument("--mbs", type=int, help="Minibatch size")
    parser.add_argument("--num_workers", type=int, help="Number of workers to use for data loading")
    parser.add_argument("--optimizer", default="adam", help="Optimizer to use. Must be one of sgd | adam | rmsprop")
    parser.add_argument("--lr", type=float, help="Learning rate to use")
    parser.add_argument("--adam_b1", type=float, default=0.9, help="b1 parameter of Adam, default=0.9")
    parser.add_argument("--adam_b2", type=float, default=0.999, help="b2 parameter of Adam, default=0.999")
    parser.add_argument(
        "--greybox",
        help="Name of greybox model. Must be one of the models provided by https://github.com/eeulig/ldct-benchmark",
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")

    # ------------------------------   Random Seeds   ------------------------------#
    parser.add_argument("--seed", default=1332, type=int, help="Random seed, default=1332")

    # -----------------------------   Method-specific   ----------------------------#
    parser = add_vae_arguments(parser)
    parser = add_cinn_arguments(parser)

    return parser


def use_config(args):
    if args.config:
        file = open(args.config)
        parsed_yaml = yaml.load(file, Loader=yaml.FullLoader)
        args = argparse.Namespace(**parsed_yaml)
    return args
