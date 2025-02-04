from dataloaders import EcgDataset1D, EcgDataset2D
import os
import os.path as osp
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
from network_utils import load_checkpoint
import models

class Runner:
    def __init__(self, config):
        self.config = config
        self.exp_name = self.config.get("exp_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.res_dir = osp.join(self.config["exp_dir"], self.exp_name, "results")
        os.makedirs(self.res_dir, exist_ok=True)

        self.model = self._init_model()
        self.inference_loader = self._init_dataloader()

        self._load_pretrained_model()

    def _init_model(self):
        model_name = self.config.get("model")

        model = getattr(models, model_name)(num_classes=self.config["num_classes"])
        return model.to(self.config["device"])

    def _init_dataloader(self):
        dataset_type = self.config.get("dataset_type", "1D")

        if dataset_type == "1D":
            dataset = EcgDataset1D(self.config["json"], self.config["mapping_json"])
        elif dataset_type == "2D":
            dataset = EcgDataset2D(self.config["json"], self.config["mapping_json"])
        else:
            raise ValueError("Unsupported dataset_type. Use '1D' or '2D'.")

        return dataset.get_dataloader(
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False
        )

    def _load_pretrained_model(self):
        pretrained_path = self.config.get("model_path")
        load_checkpoint(pretrained_path, self.model)

    def run(self):
        self.model.eval()
        gt_class, pd_class = np.array([]), np.array([])

        with torch.no_grad():
            for batch in tqdm(self.inference_loader, desc="Running Inference"):
                inputs = batch["image"].to(self.config["device"])
                predictions = self.model(inputs)
                classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

                gt_class = np.concatenate((gt_class, batch["class"].numpy()))
                pd_class = np.concatenate((pd_class, classes))

        accuracy = np.mean(pd_class == gt_class)
        print(f"Val accuracy: {accuracy:.4f}")

        np.savetxt(osp.join(self.res_dir, "predictions.txt"), pd_class.astype(int))
