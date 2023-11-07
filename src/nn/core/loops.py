import torch
import logging
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader, random_split

from .callbacks import CallBack
from src.eval.metrics import Metrics
from src.utils.misc import LogReports


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        optimizer,
        loss_fn,
        metric_list,
        batch_size,
        max_epochs,
        validation_split=-1,
        step_size=10,
        device="cpu",
        track_result_on=["mean_sqaured_loss"],
        result_directory="results/runs/",
        **kwargs,
    ):
        self.model = model
        self.ds = dataset
        self.optim = optimizer
        self.loss_fn = loss_fn
        self.bs = batch_size
        self.max_epochs = max_epochs
        self.validation_split = validation_split
        self.step_size = step_size
        self.device = (
            torch.device(device) if device == "cpu" else torch.device(f"cuda:{device}")
        )
        self.mt = Metrics(self.device)
        self.metrics = self._import_metrics(metric_list)
        self.res_dir = result_directory
        Path(self.res_dir).mkdir(parents=True, exist_ok=True)
        self.callback = CallBack(
            self.model,
            self.optim,
            self._prepare_tracker_list(track_result_on),
            f"{result_directory}/artifacts",
        )
        self.log_reports = LogReports(self.res_dir + "/logs")

    def _prepare_tracker_list(self, metric_list):
        tracker_list = []
        for metric in metric_list:
            positive_delta = "loss" in metric
            tracker_list.append((metric, positive_delta))
        return tracker_list

    def _prepare_loaders(self):
        if self.validation_split != -1:
            logging.info(
                f"Splitting the dataset into training and validation set. split_ratio={self.validation_split}"
            )
            train_size = int(len(self.ds) * self.validation_split)
            val_size = len(self.ds) - train_size
            train_ds, val_ds = random_split(self.ds, [train_size, val_size])
            train_dl = DataLoader(train_ds, batch_size=self.bs, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=self.bs, shuffle=True)
            return train_dl, val_dl
        else:
            logging.info(f"No validation set generated.")
            train_dl = DataLoader(self.ds, batch_size=self.bs, shuffle=True)
            return train_dl, None

    def _import_metrics(self, metric_list):
        metrics = {}
        for name in metric_list:
            metrics[name] = getattr(self.mt, name)
        return metrics

    def _process_metric_results(self, pred, true, thresh=0.6):
        loss = self.loss_fn(pred, true)
        pred = (pred >= thresh).int()
        metric_dict = {}
        for name, metric in self.metrics.items():
            res_value = metric(pred, true)
            metric_dict[name] = (
                res_value if not torch.isnan(res_value) else torch.float(0)
            ).item()
        return loss, metric_dict

    def _parse_single_step(self, batch, training=False):
        image_batch, mask_batch = batch
        image_batch = image_batch.to(self.device)
        mask_batch = mask_batch.to(self.device).float()
        out_ = self.model(image_batch)
        loss, metrics = self._process_metric_results(out_, mask_batch)
        metrics["loss"] = loss.item()
        if training:
            loss.backward()
            self.optim.step()
        return metrics

    def _parse_single_epoch(self, loader, training=False):
        epoch_results = []
        for ind, batch in enumerate(loader):
            metrics = self._parse_single_step(batch, training)
            epoch_results.append(metrics)
            if ind == 0 or (ind + 1) % self.step_size == 0:
                dataframe = pd.DataFrame(metrics, index=["0"])
                dataframe.insert(0, "STEP", [ind + 1])
                logging.info(f"\n{dataframe.head()}")
        epoch_results = pd.DataFrame.from_records(
            epoch_results, index=[str(ind + 1) for ind in range(len(epoch_results))]
        )
        return epoch_results

    def train(self):
        logging.info(f"Preparing Batched Data.")
        train_dl, val_dl = self._prepare_loaders()
        logging.info(f"Data Prepared.")
        logging.info(f"Loading model into device. DEVICE={self.device}")
        self.model.to(self.device)
        for epoch in range(1, self.max_epochs + 1):
            logging.info(f"Training on Epoch: {epoch}")
            epoch_res_train = self._parse_single_epoch(train_dl, training=True)
            logging.info(f"\n[Epoch/{epoch}/TRAINING] : \n{epoch_res_train.mean()}")
            epoch_res_val = self._parse_single_epoch(val_dl)
            logging.info(f"\n[Epoch/{epoch}/VALIDATION] : \n{epoch_res_val.mean()}")
            self.log_reports.update_logs(epoch_res_train, epoch_res_val)
            resume_flag = self.callback.update(self.model, self.optim, epoch_res_val)
            if not resume_flag:
                logging.info("Training stopped due to continuous degradation.")
                break
        logging.info("Training finished.")
        latest_artifact_path = self.res_dir + "/artifacts/latest_model.pt"
        latest_artifact = {
            "network": self.model.state_dict(),
            "optim": self.optim.state_dict(),
        }
        logging.info(f"Storing latest model at {latest_artifact_path}")
        torch.save(latest_artifact, latest_artifact_path)
        self.log_reports.dump_logs()
