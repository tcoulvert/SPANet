"""
Training script for Resonance Regression model.

Usage:
    python -m spanet.train_resonance_regression \
        -ef event.yaml \
        -tf training.h5 \
        -of options.json \
        -n my_experiment
"""
from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import shutil
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    ModelSummary,
    TQDMProgressBar,
    EarlyStopping
)

from spanet.options import Options
from spanet.network.resonance_regression import ResonanceRegressionModel


def main(
    event_file: str,
    training_file: str,
    validation_file: str,
    options_file: Optional[str],
    checkpoint: Optional[str],
    state_dict: Optional[str],
    log_dir: str,
    name: str,
    fp16: bool,
    verbose: bool,
    gpus: Optional[int],
    epochs: Optional[int],
    batch_size: Optional[int],
    limit_dataset: Optional[float],
    early_stopping: Optional[int],
    classify: bool,
):
    # Whether this is the master process
    master = "NODE_RANK" not in environ

    # Create options and load from file if provided
    options = Options(event_file, training_file, validation_file)

    if options_file is not None:
        with open(options_file, 'r') as f:
            options.update_options(json.load(f))

    # Command line overrides
    options.verbose_output = verbose

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding batch size: {batch_size}")
        options.batch_size = batch_size

    if limit_dataset is not None:
        if master:
            print(f"Overriding dataset limit: {limit_dataset}%")
        options.dataset_limit = limit_dataset / 100

    if epochs is not None:
        if master:
            print(f"Overriding epochs: {epochs}")
        options.epochs = epochs

    # Classification mode configuration
    # Classifications are now read directly from the dataset (CLASSIFICATIONS section in HDF5)
    if classify:
        if master:
            print("Enabling classification mode")
            print("Classification targets will be read from dataset CLASSIFICATIONS section")

        # Enable classification loss if not already set
        if options.classification_loss_scale == 0:
            options.classification_loss_scale = 1.0
            if master:
                print(f"Setting classification_loss_scale to 1.0")

    # Display configuration
    if master:
        options.display()

    # Create model
    model = ResonanceRegressionModel(options)

    # Load state dict if provided
    if state_dict is not None:
        if master:
            print(f"Loading state dict from: {state_dict}")
        state = torch.load(state_dict, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if master:
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

    # Setup logger
    log_dir = getcwd() if log_dir is None else log_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=name)

    # Setup callbacks - use different metrics for classification vs regression mode
    #
    # IMPORTANT:
    # - Avoid putting metric names with "/" into checkpoint filenames, otherwise Lightning will
    #   interpret them as directories (e.g. "loss/total_loss" -> "loss/total_loss=...ckpt"),
    #   creating lots of nested/empty folders.
    if classify:
        checkpoint_monitor = "validation_accuracy"
        checkpoint_mode = "max"
        checkpoint_filename = "epoch_{epoch}_val_acc_{validation_accuracy:.4f}"
    else:
        checkpoint_monitor = "validation_mae"
        checkpoint_mode = "min"
        checkpoint_filename = "epoch_{epoch}_val_mae_{validation_mae:.4f}"

    callbacks = [
        ModelCheckpoint(
            verbose=options.verbose_output,
            filename=checkpoint_filename,
            monitor=checkpoint_monitor,
            save_top_k=3,
            mode=checkpoint_mode,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=2) if _RICH_AVAILABLE else ModelSummary(max_depth=2)
    ]

    if early_stopping is not None and early_stopping > 0:
        callbacks.append(EarlyStopping(
            monitor=checkpoint_monitor,
            patience=early_stopping,
            mode=checkpoint_mode,
            verbose=True
        ))

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if options.num_gpu > 0 else "auto",
        devices=options.num_gpu if options.num_gpu > 0 else "auto",
        strategy="ddp" if options.num_gpu > 1 else "auto",
        precision="16-mixed" if fp16 else "32-true",
        gradient_clip_val=options.gradient_clip if options.gradient_clip > 0 else None,
        max_epochs=options.epochs,
        logger=logger,
        callbacks=callbacks
    )

    # Save configuration
    if master:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.log_dir, exist_ok=True)

        with open(f"{trainer.logger.log_dir}/options.json", 'w') as f:
            json.dump(options.__dict__, f, indent=4)

        shutil.copy2(options.event_info_file, f"{trainer.logger.log_dir}/event.yaml")

    # Train
    trainer.fit(model, ckpt_path=checkpoint)


if __name__ == '__main__':
    parser = ArgumentParser(description="Train Resonance Regression Model")

    parser.add_argument("-ef", "--event_file", type=str, default="",
                        help="Event YAML file with input/regression definitions")

    parser.add_argument("-tf", "--training_file", type=str, default="",
                        help="HDF5 file with training data")

    parser.add_argument("-vf", "--validation_file", type=str, default="",
                        help="HDF5 file with validation data (optional)")

    parser.add_argument("-of", "--options_file", type=str, default=None,
                        help="JSON file with options")

    parser.add_argument("-cf", "--checkpoint", type=str, default=None,
                        help="Checkpoint to resume training from")

    parser.add_argument("-sf", "--state_dict", type=str, default=None,
                        help="Load only model weights from checkpoint")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for logs/checkpoints")

    parser.add_argument("-n", "--name", type=str, default="resonance_regression",
                        help="Experiment name")

    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Override number of epochs")

    parser.add_argument("-g", "--gpus", type=int, default=None,
                        help="Override GPU count")

    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size")

    parser.add_argument("-p", "--limit_dataset", type=float, default=None,
                        help="Limit dataset to first L percent (0-100)")

    parser.add_argument("-es", "--early_stopping", type=int, default=None,
                        help="Early stopping patience (epochs)")

    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use mixed precision training")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Verbose output")

    parser.add_argument("--classify", action="store_true",
                        help="Enable classification mode. Classification targets are read from "
                             "the dataset CLASSIFICATIONS section. Requires classification_loss_scale > 0.")

    main(**parser.parse_args().__dict__)
