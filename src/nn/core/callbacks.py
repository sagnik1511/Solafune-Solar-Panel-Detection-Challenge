import torch
import logging
from pathlib import Path


class CallBack:
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim,
        tracker_list,
        artifact_directory,
        stop_epoch=5,
    ):
        self.best_state_dict = {
            "network": model.state_dict(),
            "optim": optim.state_dict(),
        }
        self.optim = optim
        self.tracker_list = tracker_list
        self.stop_epoch = stop_epoch
        self.res_dir = artifact_directory
        Path(self.res_dir).mkdir(parents=True, exist_ok=True)
        self.best_res = self._prepare_best_results_dict()

    def _prepare_best_results_dict(self):
        res_dict = {}
        for metric, delta in self.tracker_list:
            res_dict[metric] = float("inf") if delta else 0
        return res_dict

    def update(self, model, optim, epoch_res, precision=6):
        pass_flag = False
        for metric, delta in self.tracker_list:
            updated = (
                round(self.best_res[metric], precision)
                > round(epoch_res[metric].mean(), precision)
                if delta
                else round(self.best_res[metric], precision)
                < round(epoch_res[metric].mean(), precision)
            )
            if updated:
                logging.info(
                    f"Model has updated on {metric}, {self.best_res[metric]} -> {epoch_res[metric].mean()}"
                )
            # else:
            #     logging.info(
            #         f"Model hasn't updated on {metric}, {self.best_res[metric]} -> {epoch_res[metric].mean()}"
            #     )
            pass_flag |= updated
        if pass_flag:
            self.best_res = epoch_res.mean().to_dict()
            self.best_state_dict = {
                "network": model.state_dict(),
                "optim": optim.state_dict(),
            }
            logging.info(f"Updating best model info at {self.res_dir}/best_model.pt")
            torch.save(self.best_state_dict, f"{self.res_dir}/best_model.pt")
        else:
            logging.info("Model not updated.")
            self.stop_epoch -= 1
        return self.stop_epoch
