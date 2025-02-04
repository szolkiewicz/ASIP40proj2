from dataloaders import EcgDataset1D, EcgDataset2D
from torch.utils.tensorboard import SummaryWriter
import models
from datetime import datetime
import os.path as osp
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from network_utils import save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, config):
        self.config = config
        self.exp_name = self.config.get("exp_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_dir = osp.join(self.config["exp_dir"], self.exp_name, "logs")
        self.pth_dir = osp.join(self.config["exp_dir"], self.exp_name, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.pth_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.criterion = nn.CrossEntropyLoss().to(self.config["device"])

        self.train_loader, self.val_loader = self._init_dataloaders()

        self._load_pretrained_model()
        self.epochs = self.config.get("epochs", int(1e5))

    def _init_model(self):
        model_name = self.config.get("model")

        model = getattr(models, model_name)(num_classes=self.config["num_classes"])
        return model.to(self.config["device"])

    def _init_dataloaders(self):
        dataset_type = self.config.get("dataset_type", "1D")

        if dataset_type == "1D":
            train_dataset = EcgDataset1D(self.config["train_json"], self.config["mapping_json"])
            val_dataset = EcgDataset1D(self.config["val_json"], self.config["mapping_json"])
        elif dataset_type == "2D":
            train_dataset = EcgDataset2D(self.config["train_json"], self.config["mapping_json"])
            val_dataset = EcgDataset2D(self.config["val_json"], self.config["mapping_json"])
        else:
            raise ValueError("Unsupported dataset_type. Use '1D' or '2D'.")

        train_loader = train_dataset.get_dataloader(
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
        )
        val_loader = val_dataset.get_dataloader(
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
        )

        return train_loader, val_loader

    def _init_optimizer(self):
        optimizer = getattr(optim, self.config["optim"])(
            self.model.parameters(), **self.config["optim_params"]
        )
        return optimizer

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        gt_class = np.empty(0)
        pd_class = np.empty(0)

        for i, batch in enumerate(self.train_loader):
            inputs = batch["image"].to(self.config["device"])
            targets = batch["class"].to(self.config["device"])

            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()
            gt_class = np.concatenate((gt_class, batch["class"].numpy()))
            pd_class = np.concatenate((pd_class, classes))

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"\tIter [{i + 1}/{len(self.train_loader)}] Loss: {loss.item():.4f}")

            self.writer.add_scalar("Train loss (iterations)", loss.item(), self.total_iter)
            self.total_iter += 1

        total_loss /= len(self.train_loader)
        class_accuracy = np.mean(pd_class == gt_class)

        print(f"Train loss - {total_loss:.4f}")

        self.writer.add_scalar("Train loss (epochs)", total_loss, self.training_epoch)

    def _load_pretrained_model(self):
        pretrained_path = self.config.get("model_path")
        if pretrained_path:
            self.training_epoch, self.total_iter = load_checkpoint(
                pretrained_path, self.model, optimizer=self.optimizer
            )
        else:
            self.training_epoch = 0
            self.total_iter = 0

    def val(self):
        self.model.eval()
        total_loss = 0
        gt_class = np.empty(0)
        pd_class = np.empty(0)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = batch["image"].to(self.config["device"])
                targets = batch["class"].to(self.config["device"])

                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)

                classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()
                gt_class = np.concatenate((gt_class, batch["class"].numpy()))
                pd_class = np.concatenate((pd_class, classes))

                total_loss += loss.item()

        total_loss /= len(self.val_loader)
        class_accuracy = np.mean(pd_class == gt_class)

        print(f"Val loss - {total_loss:.4f}")

        self.writer.add_scalar("Val loss", total_loss, self.training_epoch)

    def train(self):
        for epoch in range(self.training_epoch, self.epochs):
            print(f"Epoch - {self.training_epoch + 1}")
            self.train_epoch()
            save_checkpoint(
                {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "total_iter": self.total_iter,
                },
                osp.join(self.pth_dir, f"{epoch:08d}.pth"),
            )
            self.val()
            self.training_epoch += 1
