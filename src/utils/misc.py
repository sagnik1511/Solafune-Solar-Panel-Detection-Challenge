import logging
import pandas as pd
from pathlib import Path


class LogReports:
    def __init__(self, log_directory):
        self.training_logs = []
        self.validation_logs = []
        Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.train_log_path = log_directory + "/train.csv"
        self.val_log_path = log_directory + "/val.csv"

    def update_logs(self, train_res, val_res):
        self.training_logs.append(train_res.mean())
        self.validation_logs.append(val_res.mean())

    def dump_logs(self):
        train_log_df = pd.DataFrame.from_records(self.training_logs)
        val_log_df = pd.DataFrame.from_records(self.validation_logs)
        logging.info(f"Writing training results to {self.train_log_path}")
        train_log_df.to_csv(self.train_log_path)
        logging.info(f"Writing validation results to {self.val_log_path}")
        val_log_df.to_csv(self.val_log_path)
