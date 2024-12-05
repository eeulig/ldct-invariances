import glob
import importlib
import os
import shutil
import time
import warnings

import matplotlib
import numpy as np
import torch

import ldctinv.utils.auxiliaries as aux
from argparser import make_parser, use_config

torch.autograd.set_detect_anomaly(True)
import wandb

matplotlib.use("Agg")

os.environ["WANDB_START_METHOD"] = "thread"  # Necessary to spawn subprocess on cluster node
os.environ["WANDB_IGNORE_GLOBS"] = "*.pt"  # Do not upload network models to wandb to save space


def train(args):
    # Setup seeds
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device(s) to use
    if isinstance(args.devices, list) and len(args.devices) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.devices])
        args.devices = list(range(len(args.devices)))
    elif isinstance(args.devices, list) and len(args.devices) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[0])
        args.devices = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
        args.devices = 0
    device = torch.device("cuda" if args.cuda else "cpu")

    while True:
        try:
            wandb.init(project="ldct-invariances", config=args)
            break
        except Exception as e:
            print(f"{e} ... retrying...")
            time.sleep(10)

    wandb.run.name = wandb.run.dir.split("/")[-2].split("run-")[-1]
    # Save all python files in wandb. Exclude the ones saved in run dirs
    for fn in glob.glob("**/*.py", recursive=True):
        if "wandb" not in os.path.dirname(fn) and "env" not in os.path.dirname(fn):
            os.makedirs(
                os.path.dirname(os.path.join(wandb.run.dir, fn)), exist_ok=True
            )  # Create dir if not already existing
            shutil.copy(fn, os.path.join(wandb.run.dir, fn), follow_symlinks=False)
    wandb.save()
    aux.dump_config(args, wandb.run.dir)

    # Setup trainer
    try:
        trainer_module = importlib.import_module("ldctinv.{}.Trainer".format(args.trainer))
    except ModuleNotFoundError:
        raise ValueError("Trainer {} not known and module ldctinv.{}.Trainer not found".format(args.trainer))
    trainer_class = getattr(trainer_module, "Trainer")
    trainer = trainer_class(args, device)

    # Train model
    print("Start training...")
    trainer.fit()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    args = use_config(args)
    if not hasattr(args, "datafolder") or not args.datafolder:
        if "LDCT_DATA" in os.environ:
            args.datafolder = os.environ["LDCT_DATA"]
            warnings.warn(
                f"No datafolder in args. Will use the one provided via environment variable LDCT_DATA: {args.datafolder}"
            )
        else:
            raise ValueError(
                "No datafolder provided! Add via\n \t- Config file: add key: datafolder\n\t- Arguments: add argument --datafolder\n\t- Environment variable: export LDCT_DATA=..."
            )

    if args.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"

    train(args)
