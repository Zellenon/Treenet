#!/usr/bin/env python3

from simple_term_menu import TerminalMenu
from yaml import safe_load


def select_path(parent, glob):
    from pathlib import (Path)
    items = [str(w) for w in parent.glob(glob)]
    terminal_menu = TerminalMenu(items)
    return Path(items[terminal_menu.show()])


def select_yaml(dir):
    from pathlib import (Path)
    selection = select_path(dir, "*.yaml")
    with open(Path(selection)) as f:
        params = safe_load(f)
    return params


def new_dataset():
    from db import add_database
    from global_config import (
        DatasetConfig,
        dataset_config_dir,
    )

    params = DatasetConfig(select_yaml(dataset_config_dir))
    replace_behaviors = ["Overwrite existing entries", "Skip existing entries"]
    replace_choice = TerminalMenu(replace_behaviors).show()
    add_database(params, replace=replace_choice == 0)


def pre_process():
    from global_config import (
        DatasetConfig,
        dataset_config_dir,
    )
    params = DatasetConfig(select_yaml(dataset_config_dir))

    from preprocess import process

    process(params, vocab_size=500000, max_len=500)


def train():
    from global_config import (
        DatasetConfig,
        dataset_config_dir,
        model_config_dir,
    )
    dataset_params = DatasetConfig(select_yaml(dataset_config_dir))
    model_params = select_yaml(model_config_dir)
    refiner = ["None", "CorNet", "TreeNet"]
    refiner_choice = TerminalMenu(refiner).show()

    from train import train_model

    train_model(dataset_params, model_params, refiner[refiner_choice])


def evaluate():
    pass


DB = "Load New Dataset"
PRE = "Pre-process Database"
TRAIN = "Train Model with Dataset"
EVAL = "Evaluate Trained Model"
main_menu_exits = {DB: new_dataset, PRE: pre_process, TRAIN: train, EVAL: evaluate}

terminal_menu = TerminalMenu([DB, PRE, TRAIN, EVAL])
main_menu_choice = terminal_menu.show()
main_menu_exits[[DB, PRE, TRAIN, EVAL][main_menu_choice or 0]]()
