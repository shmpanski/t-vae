import argparse
import logging
import os

import torch
import yaml
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from tvae.data import TVAEDataset
from tvae.models import TransformerVAE

WORKBENCH_DIR = "./workbench"
DEVELOP_MODE = False


class Trainer():
    def __init__(self, config):
        self.config = yaml.load(config)
        self.targs = self.config["training"]["args"]
        self.margs = self.config["model"]["args"]
        self.optim_name = self.config["optimizer"]["name"]
        self.oargs = self.config["optimizer"]["args"]
        self.dargs = self.config["dataset"]["args"]

        self.dump_directory = os.path.join(WORKBENCH_DIR, self.targs["prefix"])
        self.tensorboard_log_dir = os.path.join(WORKBENCH_DIR, "logs", self.targs["prefix"])
        logging.info("Configuration `%s` with launch prefix `%s` have been loaded", config.name, self.targs["prefix"])

        self.train_dataset, self.test_dataset = self.load_dataset()
        logging.info("Dataset loaded. Train part size: %d. Test part size: %d",
                     len(self.train_dataset), len(self.test_dataset))

        self.device = torch.device(self.targs["device"])
        logging.info("Device `%s` selected for training", self.device)

        self.model, self.model_args = TransformerVAE.create(self.train_dataset, self.margs)
        self.model.to(self.device)
        logging.info("Model loaded.\nArguments: %s.\nTotal parameters: %d ",
                     str(self.model_args),
                     sum(p.numel() for p in self.model.learnable_parameters()))

        self.optimizer = getattr(torch.optim, self.optim_name)(self.model.learnable_parameters(), **self.oargs)
        logging.info("Create `%s` optimizer with parameters: %s", self.optim_name, self.oargs)

        self.tb_writer = SummaryWriter(self.tensorboard_log_dir)
        self.pbar_descr = "Epoch[{}] | Loss[{:.2f}]"
        self.pbar = tqdm(initial=0, leave=False, desc=self.pbar_descr)

    def run(self):
        trainer = self.model.create_trainer(self.optimizer, self.device)
        train_loader = DataLoader(self.train_dataset,
                                  self.targs.get("train_batch_size", 8),
                                  shuffle=True,
                                  collate_fn=self.train_dataset.collate_function)
        trainer_saver = ModelCheckpoint(self.dump_directory,
                                        filename_prefix="checkpoint",
                                        save_interval=self.targs.get("checkpoint_interval", 1000),
                                        n_saved=1,
                                        save_as_state_dict=True,
                                        create_dir=True,
                                        require_empty=False)

        to_save = {"model": self.model, "optimizer": self.optimizer}

        # Create evaluator engine
        evaluator = self.model.create_evaluator(self.device)
        val_loader = DataLoader(self.test_dataset,
                                self.targs.get("test_batch_size", 8),
                                shuffle=True,
                                collate_fn=self.test_dataset.collate_function,
                                drop_last=True)

        trainer.add_event_handler(Events.ITERATION_COMPLETED, trainer_saver, to_save)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self.get_train_logger(train_loader))
        trainer.add_event_handler(Events.EPOCH_STARTED, self.get_pbar_initializer(train_loader))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.get_pbar_destructor())
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: evaluator.run(val_loader))
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, self.get_sampler())

        trainer.run(train_loader, self.targs.get("epochs", 1))
        self.tb_writer.close()
        self.pbar.close()

    def load_dataset(self):
        train_part_name = "develop" if DEVELOP_MODE else "train"
        train_dataset = TVAEDataset(part=train_part_name, **self.dargs)
        test_dataset = TVAEDataset(part="test", spm_model=train_dataset.spm_model, **self.dargs)

        return train_dataset, test_dataset

    def get_train_logger(self, train_loader: DataLoader):
        def _logger(engine: Engine):
            epoch = engine.state.epoch
            iteration = (engine.state.iteration - 1) % len(train_loader) + 1
            loss = engine.state.loss
            kld = engine.state.kld

            if iteration % 10 == 0:
                self.pbar.desc = self.pbar_descr.format(epoch, loss)
                self.pbar.update(10)

            if iteration % self.targs.get("train_log_interval", 100) == 0:
                message = "Epoch[{}] | Iteration[{}/{}] | Loss: {:.4f} KLD: {:.4f}"
                message = message.format(epoch, iteration, len(train_loader), loss, kld)
                logging.info(message)
                self.tb_writer.add_scalars("training", dict(loss=loss, kld=kld), engine.state.iteration)
        return _logger

    def get_sampler(self):
        def _sample(engine: Engine):
            logging.info("ENTER")
            generated_distr, target = engine.state.output
            generated_seq = engine.state.generated_seq
            target_str = self.train_dataset.decode(target)
            generated_str = self.train_dataset.decode(generated_seq)
            for i, d in enumerate(zip(target_str, generated_str)):
                original, generated = d
                logging.info("Evaluation sample: \nOriginal sequence: %s\nDecoded sequence: %s",
                             original, generated)
            engine.terminate_epoch()
        return _sample

    def get_pbar_initializer(self, loader):
        def _pbar_initializer(enigine: Engine):
            self.pbar.total = len(loader)
            self.pbar.unpause()
        return _pbar_initializer

    def get_pbar_destructor(self):
        def _pbar_destructor(engine: Engine):
            self.pbar.total = None
            self.pbar.n = self.pbar.last_print_n = 0
            self.pbar.desc = "Epoch[{}] | Evaluating".format(engine.state.epoch)
        return _pbar_destructor


def setup_logging(default_path='logging.yml', default_level=logging.INFO, env_key='LOG_CFG'):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", nargs=1, type=str, help="runinig configuration file")
    parser.add_argument("--d", action="store_true", help="enable develop mode")
    args = parser.parse_args()

    DEVELOP_MODE = args.d

    config_filename = args.config_file[0]
    with open(config_filename, "r") as config_file:
        logging.info("Loaded configurations from %s", config_file.name)
        trainer = Trainer(config_file)
        trainer.run()
