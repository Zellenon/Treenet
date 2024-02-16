#!/usr/bin/env python3

from consolemenu import ConsoleMenu, MultiSelectMenu
from consolemenu.items import SubmenuItem
from consolemenu.items import FunctionItem
from yaml import safe_load

from global_config import AppConfig, result_log_dir

app_config = AppConfig()

def dataset_selector():
    menu = MultiSelectMenu(
        "Dataset Selection",
        app_config.get_datasets,
        epilogue_text=("Please select one or more entries separated by commas, and/or a range "
                       "of numbers. For example:  1,2,3   or   1-4   or   1,3-4"),
        exit_option_text="Back to Main",
    )

    for path, name in app_config.all_datasets.items():
        menu.append_item(FunctionItem(name, app_config.add_dataset, args=[path]))
    return menu


def model_selector():
    from global_config import model_config_dir

    menu = MultiSelectMenu(
        "Model Selection",
        app_config.get_models,
        epilogue_text=("Please select one or more entries separated by commas, and/or a range "
                       "of numbers. For example:  1,2,3   or   1-4   or   1,3-4"),
        exit_option_text="Back to Main",
    )

    for path, name in app_config.all_models.items():
        menu.append_item(FunctionItem(name, app_config.add_model, args=[path]))
    return menu


def refiner_selector():
    menu = MultiSelectMenu(
        "Refiner Selection",
        app_config.get_refiners,
        epilogue_text=("Please select one or more entries separated by commas, and/or a range "
                       "of numbers. For example:  1,2,3   or   1-4   or   1,3-4"),
        exit_option_text="Back to Main",
    )

    for r in app_config.selected_refiners.keys():
        menu.append_item(FunctionItem(r, app_config.toggle_refiner, args=[r]))
    return menu


def task_selector():
    menu = MultiSelectMenu(
        "Task Selection",
        app_config.get_tasks,
        epilogue_text=("Please select one or more entries separated by commas, and/or a range "
                       "of numbers. For example:  1,2,3   or   1-4   or   1,3-4"),
        exit_option_text="Back to Main",
    )

    for k, v in app_config.selected_tasks.items():
        menu.append_item(FunctionItem(app_config.task_names[k], app_config.toggle_task, args=[k]))
    return menu


def pre_process(dataset_cfg_path, _, _2):
    from global_config import DatasetConfig
    from preprocess import process

    process(DatasetConfig(safe_load(open(dataset_cfg_path))),
            vocab_size=500000,
            max_len=500)


def train(dataset_cfg_path, model_cfg_path, refiner):
    from global_config import DatasetConfig
    from train import train_model

    dataset_params = DatasetConfig(safe_load(open(dataset_cfg_path)))
    model_params = safe_load(open(model_cfg_path))

    # train_model(dataset_params, model_params, refiner[refiner_choice])
    train_model(dataset_params, model_params, refiner)


def test(dataset_cfg_path, model_cfg_path, refiner):
    from global_config import DatasetConfig
    from test import predict
    dataset_params = DatasetConfig(safe_load(open(dataset_cfg_path)))
    model_params = safe_load(open(model_cfg_path))
    predict(dataset_params, model_params, refiner)


def evaluate(dataset_cfg_path, model_cfg_path, refiner):
    from global_config import DatasetConfig
    from evaluation import evaluate

    dataset_params = DatasetConfig(safe_load(open(dataset_cfg_path)))
    model_params = safe_load(open(model_cfg_path))

    results = evaluate(dataset_params, model_params, refiner)

    with open(
                  result_log_dir / model_params["name"] + "-" + dataset_params.name + "-" +
                                     refiner + ".txt", 'w'
                                     ) as logfile:
        logfile.write("\n\n".join([f"{secname}:\n" +"\n".join(
            [f"{metric}: {score}" for metric,score in sec.items()]
            ) for secname, sec in results.items()]))


app_config.tasks = {
    "PRE": pre_process,
    "TRAIN": train,
    "TEST": test,
    "EVAL": evaluate
}

if __name__ == "__main__":
    menu = ConsoleMenu("Main Menu", app_config.all_info, exit_option_text="Execute Tasks")

    dataset_menu = SubmenuItem("Select Datasets", dataset_selector())
    dataset_menu.set_menu(menu)
    menu.append_item(dataset_menu)

    model_menu = SubmenuItem("Select Models", model_selector())
    model_menu.set_menu(menu)
    menu.append_item(model_menu)

    refiner_menu = SubmenuItem("Select Refiners", refiner_selector())
    refiner_menu.set_menu(menu)
    menu.append_item(refiner_menu)

    task_menu = SubmenuItem("Select Tasks", task_selector())
    task_menu.set_menu(menu)
    menu.append_item(task_menu)
    menu.start()
    menu.join()
    app_config.exec()

