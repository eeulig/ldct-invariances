import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from ldctbench.hub import load_model
from torch.autograd import Variable
from tqdm import tqdm

from ldctinv import utils
from ldctinv.cinn.network import ConditionalTransformer
from ldctinv.data.LDCTMayo import LDCTMayo


class Trainer(object):
    def __init__(self, args, device):
        self.args = args
        self.dev = device

        # Setup datasets
        self.data = {phase: LDCTMayo(phase, self.args) for phase in ["train", "val"]}
        self.dataloader = utils.setup_dataloader(self.args, self.data)

        # Setup loss
        self.criterion = utils.UnsupervisedINNLoss()

        # Setup models
        self.ae, ae_args = utils.setup_trained_model(
            args.vae,
            self.dev,
            return_args=True,
            network_name="BigAE",
            state_dict="generator",
            in_ch=2,
            out_ch=1,
            cond_ch=1,
        )  # setup conditioned vae
        self.ae.to(self.dev)
        self.ae_domain = ae_args.domain
        greybox = ae_args.greybox

        self.greybox = load_model(greybox).to(self.dev)  # setup greybox model
        self.model = ConditionalTransformer(args).to(self.dev)  # setup cinn

        if isinstance(self.args.devices, list):
            self.ae = nn.DataParallel(self.ae, device_ids=self.args.devices)
            self.model = nn.DataParallel(self.model, device_ids=self.args.devices)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.adam_b1, args.adam_b2))

        # Setup logging
        self.losses = utils.metrics.Losses(self.dataloader, ["cinn loss"])
        self.savedir = wandb.run.dir
        self.epoch = 0

    def train_step(self, batch):
        xin, y = batch["x"], batch["y"]
        with torch.no_grad():
            y_hat = self.greybox(xin)
            ae_inp = torch.cat([xin if self.ae_domain == "x" else y, y_hat], 1)
            zae = self.ae.encode(ae_inp).sample()

        zz, logdet = self.model(zae, y_hat)
        loss = self.criterion(zz, logdet)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.push(loss={"cinn loss": loss.data}, phase="train")

    def train(self):
        self.model.train()
        for batch in tqdm(self.dataloader["train"]):
            batch = {k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()}
            self.train_step(batch)

        self.losses.summarize("train")

    @torch.no_grad()
    def val_step(self, batch_idx, batch):
        xin, y = batch["x"], batch["y"]
        y_hat = self.greybox(xin)
        ae_inp = torch.cat([xin if self.ae_domain == "x" else y, y_hat], 1)
        zae = self.ae.encode(ae_inp).sample()
        zz, logdet = self.model(zae, y_hat)

        loss = self.criterion(zz, logdet)
        self.losses.push(loss={"cinn loss": loss.data}, phase="val")

        # Plot samples
        if batch_idx < self.args.loginterv:
            zz_sample = torch.randn_like(zz)
            zae_sample = self.model.reverse(zz_sample, y_hat)
            xae_sample = self.ae.decode(zae_sample, im_cond=y_hat)
            self.log_wandb_images({"input": xin, "conditioning_vae": y_hat, "reconstructions": xae_sample})

    def validate(self):
        self.images = {}
        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(self.dataloader["val"])):
            batch = {k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()}
            self.val_step(batch_idx, batch)

        self.losses.summarize("val")

    def log_wandb_images(self, images):
        for tag, img in images.items():
            img = wandb.Image(img.data.cpu()[0], caption=tag)
            if tag not in self.images:
                self.images[tag] = [img]
            else:
                self.images[tag].append(img)

    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.savedir, "cinn.pt")
        state_dict = self.model.module.state_dict() if isinstance(self.args.devices, list) else self.model.state_dict()
        torch.save({"epoch": self.epoch, "model_state_dict": state_dict}, checkpoint_path)

    def log(self):
        self.save_checkpoint()
        self.losses.log(self.savedir, self.epoch)
        self.losses.plot(self.savedir)
        wandb.log(self.images, step=self.epoch + 1)

    def fit(self):
        for epoch in range(self.args.epochs):
            # Set manual seed
            torch.manual_seed(self.args.seed + epoch)

            # Train and validate
            self.train()
            self.validate()

            # Log
            self.log()

            # Prepare next iteration
            self.losses.reset()
            self.epoch += 1
