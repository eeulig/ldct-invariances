import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from ldctbench.hub import load_model
from torch.autograd import Variable
from tqdm import tqdm

import ldctinv.utils as utils
from ldctinv.data.LDCTMayo import LDCTMayo
from ldctinv.vae.critic import Critic
from ldctinv.vae.network import BigAE


class Trainer(object):
    def __init__(self, args, device):
        self.args = args
        self.dev = device

        # Setup datasets
        self.data = {phase: LDCTMayo(phase, self.args) for phase in ["train", "val"]}
        self.dataloader = utils.setup_dataloader(self.args, self.data)

        # Setup loss
        self.criterion = utils.AELoss(self.args)
        self.perceptual = utils.PerceptualLoss(network="vgg16", device=self.dev, in_ch=1)

        if self.args.cond_critic and not self.args.conditional:
            raise ValueError("Conditional critic requires a conditional model")

        if self.args.conditional:
            if not args.greybox:
                raise ValueError("Conditional training requires a greybox model")
            # Setup greybox model
            self.greybox = load_model(args.greybox).to(self.dev)

        # Setup model and its optimizer
        self.cond_ch = 1 if args.conditional else 0
        self.generator = BigAE(self.args, in_ch=1 + self.cond_ch, out_ch=1, cond_ch=self.cond_ch).to(self.dev)
        self.critic = Critic(
            self.args, in_ch=2 if self.args.cond_critic else 1, input_ft=64, depth=5, max_ft=512, norm="in"
        ).to(self.dev)

        if isinstance(self.args.devices, list):
            self.generator = nn.DataParallel(self.generator, device_ids=self.args.devices)
            self.critic = nn.DataParallel(self.critic, device_ids=self.args.devices)

        self.g_optimizer = utils.setup_optimizer(args, self.generator.parameters())
        self.d_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr * 4, betas=(args.adam_b1, args.adam_b2))

        # Setup logging
        self.losses = utils.metrics.Losses(
            self.dataloader, ["D loss", "G loss adv", "G loss ae", "grad. pen.", "G loss perc.", "G loss rec."]
        )
        self.savedir = wandb.run.dir

        self.epoch = 0

    def gradient_penalty(self, target, fake, lam=10):
        assert target.size() == fake.size()
        a = torch.FloatTensor(np.random.random((target.size(0), 1, 1, 1))).to(self.dev)
        interp = (a * target + ((1 - a) * fake)).requires_grad_(True)
        d_interp = self.critic(interp)
        fake_ = torch.FloatTensor(target.shape[0], 1).fill_(1.0).to(self.dev).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lam
        return gradient_penalty

    def train_step(self, batch):
        xin, y = batch["x"], batch["y"]
        inputs = batch[self.args.domain]
        targets = inputs.detach().clone()

        if self.args.conditional:
            with torch.no_grad():
                im_cond = self.greybox(xin).detach()
        else:
            im_cond = None

        #  Train Critic
        _, fakes = self.generator(inputs, im_cond=im_cond)
        if self.args.cond_critic:
            c_fakes = torch.cat([fakes, im_cond], dim=1)
            c_targets = torch.cat([targets, im_cond], dim=1)
        else:
            c_fakes = fakes
            c_targets = targets

        grad_p = self.gradient_penalty(c_targets, c_fakes, lam=self.args.lam)
        critic_loss = torch.mean(self.critic(c_fakes)) - torch.mean(self.critic(c_targets))
        loss_D = critic_loss + grad_p

        self.d_optimizer.zero_grad()
        loss_D.backward()
        self.d_optimizer.step()

        #  Train Generator
        self.g_optimizer.zero_grad()

        posterior, fakes = self.generator(inputs, im_cond=im_cond)
        loss_G_rec, loss_G_ae = self.criterion(fakes, targets, posterior)
        loss_G_perc = self.perceptual(fakes, targets)
        if self.args.cond_critic:
            loss_G_adv = -torch.mean(self.critic(torch.cat([fakes, im_cond], dim=1)))
        else:
            loss_G_adv = -torch.mean(self.critic(fakes))

        loss_G = (
            self.args.alpha * loss_G_rec + self.args.kl_weight * loss_G_ae + self.args.beta * loss_G_perc + loss_G_adv
        )

        loss_G.backward()
        self.g_optimizer.step()

        self.losses.push(
            loss={
                "D loss": critic_loss.data,
                "G loss adv": loss_G_adv.data,
                "G loss ae": loss_G_ae.data,
                "G loss perc.": loss_G_perc.data,
                "G loss rec.": loss_G_rec.data,
                "grad. pen.": grad_p.data,
            },
            phase="train",
        )

    @torch.no_grad()
    def val_step(self, batch_idx, batch):
        xin, y = batch["x"], batch["y"]
        inputs = batch[self.args.domain]
        targets = inputs.detach().clone()

        if self.args.conditional:
            with torch.no_grad():
                im_cond = self.greybox(xin).detach()
        else:
            im_cond = None

        posterior, reconstructions = self.generator(inputs, im_cond=im_cond)

        loss_G_rec, loss_G_ae = self.criterion(reconstructions, targets, posterior)
        loss_G_perc = self.perceptual(reconstructions, targets)

        self.losses.push(
            loss={"G loss ae": loss_G_ae.data, "G loss perc.": loss_G_perc.data, "G loss rec.": loss_G_rec}, phase="val"
        )

        if batch_idx < self.args.loginterv:
            if self.args.conditional:
                self.log_wandb_images(
                    {"inputs": inputs, "reconstructions": reconstructions, "targets": targets, "conditioning": im_cond}
                )
            else:
                self.log_wandb_images({"inputs": inputs, "reconstructions": reconstructions, "targets": targets})

    def train(self):
        self.generator.train()
        for batch_idx, batch in enumerate(tqdm(self.dataloader["train"])):
            batch = {k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()}
            self.train_step(batch)

        self.losses.summarize("train")

    def validate(self):
        self.images = dict()
        self.generator.eval()
        for batch_idx, batch in enumerate(tqdm(self.dataloader["val"])):
            batch = {k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()}
            self.val_step(batch_idx, batch)

        self.losses.summarize("val")

    def save_checkpoint(self, store_critic=False):
        checkpoint_path = os.path.join(self.savedir, "generator.pt")
        state_dict = (
            self.generator.module.state_dict() if isinstance(self.args.devices, list) else self.generator.state_dict()
        )
        torch.save({"epoch": self.epoch, "model_state_dict": state_dict}, checkpoint_path)
        if store_critic:
            checkpoint_path = os.path.join(self.savedir, "critic.pt")
            state_dict = (
                self.critic.module.state_dict() if isinstance(self.args.devices, list) else self.critic.state_dict()
            )
            torch.save({"epoch": self.epoch, "model_state_dict": state_dict}, checkpoint_path)

    def log(self):
        self.save_checkpoint(store_critic=True)
        self.losses.log(self.savedir, self.epoch)
        self.losses.plot(self.savedir)
        wandb.log(self.images, step=self.epoch + 1)

    def log_wandb_images(self, images):
        for tag, img in images.items():
            img = wandb.Image(img.data.cpu()[0], caption=tag)
            if tag not in self.images:
                self.images[tag] = [img]
            else:
                self.images[tag].append(img)

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
