import logging
import torch.nn as nn
from torch import optim
from argparse import ArgumentParser
from configparser import ConfigParser


from src.nn.core.loops import Trainer
from src.nn import networks
from src.data.dataset import Sentinel2Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
CONFIG_PATH = "config/train.cfg"


def train(config_section):
    config = ConfigParser()
    config.read(CONFIG_PATH)
    configuration = dict(config[config_section.title()].items())
    for param, field in configuration.items():
        if "," in field:
            configuration[param] = field.split(",")
        if "true" in field or "false" in field:
            configuration[param] = "true" in field
        try:
            field = int(field)
            configuration[param] = field
        except:
            try:
                field = float(field)
                configuration[param] = field
            except:
                pass
    logging.info(f"Running on Configration: {configuration}")
    dataset = Sentinel2Dataset(**configuration)
    logging.info(f"{len(dataset)} records found for training")
    model = getattr(networks, config_section.title())(accumulated=True)
    optimizer = getattr(optim, configuration["optim"])(
        params=model.parameters(), lr=1e-4
    )
    loss_fn = getattr(nn, configuration["criterion"])()
    trainer = Trainer(model, dataset, optimizer, loss_fn, **configuration)
    trainer.train()


def define_parser():
    parser = ArgumentParser(description="Training Tool")
    parser.add_argument("-s", "--section", type=str, required=True, default="baseline")
    return parser.parse_args()


def main():
    args = define_parser()
    train(args.section)


if __name__ == "__main__":
    main()
