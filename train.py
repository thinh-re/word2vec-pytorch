import os
import shutil
from typing import Dict

import torch
import torch.nn as nn
import yaml
from tap import Tap

from utils.dataloader import get_dataloader_and_vocab
from utils.helper import (get_lr_scheduler, get_model_class,
                          get_optimizer_class, save_config, save_vocab)
from utils.trainer import Trainer


def train(config: Dict):
    shutil.rmtree(config["model_dir"], ignore_errors=True)
    os.makedirs(config["model_dir"])
    
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="train",
        data_dir=config["data_dir"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None,
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="valid",
        data_dir=config["data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        vocab=vocab,
    )

    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])
    
class TrainArgumentParser(Tap):
    config: str # path to yaml config
    
if __name__ == '__main__':
    args = TrainArgumentParser().parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(config)
