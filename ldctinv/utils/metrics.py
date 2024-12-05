import csv
import os

import matplotlib.pyplot as plt
import wandb


class Losses(object):
    def __init__(self, dataloader, losses="px"):
        self.dataloader = dataloader
        self.names = losses if isinstance(losses, list) else [losses]
        self.losses = {phase: {loss: [0.0] for loss in self.names} for phase in dataloader.keys()}

    def push(self, loss, phase, name="px"):
        if isinstance(loss, dict):
            for n, l in loss.items():
                self.losses[phase][n][-1] += l.item()
        else:
            self.losses[phase][name][-1] += loss.item()

    def summarize(self, phase):
        for name in self.names:
            self.losses[phase][name][-1] /= len(self.dataloader[phase])

    def reset(self):
        for phase in self.losses:
            for name in self.names:
                self.losses[phase][name].append(0.0)

    def log(self, savepath, epoch):
        # Log to wandb
        wandb.log(
            {
                "Loss {} ({})".format(name, phase): loss[-1]
                for phase, losses in self.losses.items()
                for name, loss in losses.items()
            },
            step=epoch + 1,
        )

        # Log to file
        with open(os.path.join(savepath, "Losses.csv"), "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if epoch == 0:
                writer.writerow(
                    ["Epoch"] + ["{} {}".format(name, phase) for name in self.names for phase in self.losses]
                )
            writer.writerow([epoch] + [loss[name][-1] for name in self.names for loss in self.losses.values()])

    def plot(self, savepath, y_log=False):
        for name in self.names:
            plt.figure(figsize=(5, 3))
            for phase in self.losses.keys():
                y = self.losses[phase][name]
                x = list(range(len(y)))
                plt.plot(x, y, label=phase)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                if y_log:
                    plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, "Losses_{}.pdf".format(name)))
            plt.close("all")


class Metrics(object):
    def __init__(self, dataloader, metrics):
        self.dataloader = dataloader
        self.names = metrics if isinstance(metrics, list) else [metrics]
        self.metrics = {metric: [0.0] for metric in self.names}

    def push(self, metric, name="px"):
        if isinstance(metric, dict):
            for n, m in metric.items():
                self.metrics[n][-1] += m
        else:
            self.metrics[name][-1] += metric

    def summarize(self):
        for name in self.names:
            self.metrics[name][-1] /= len(self.dataloader["val"])

    def reset(self):
        for name in self.names:
            self.metrics[name].append(0.0)

    def log(self, savepath, epoch):
        # Log to wandb
        wandb.log({"Metric {}".format(name): metric[-1] for name, metric in self.metrics.items()}, step=epoch + 1)

        # Log to file
        with open(os.path.join(savepath, "Metrics.csv"), "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if epoch == 0:
                writer.writerow(["Epoch"] + ["{}".format(name) for name in self.names])
            writer.writerow([epoch] + [self.metrics[name][-1] for name in self.names])

    def plot(self, savepath):
        for name in self.names:
            plt.figure(figsize=(5, 3))
            y = self.metrics[name]
            x = list(range(len(y)))
            plt.plot(x, y, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, "Metrics_{}.pdf".format(name)))
            plt.close("all")
