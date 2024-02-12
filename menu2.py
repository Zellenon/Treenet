#!/usr/bin/env python3

from consolemenu import ConsoleMenu, MultiSelectMenu
from consolemenu.items import MenuItem, SubmenuItem
from consolemenu.items import FunctionItem

from yaml import safe_load

PRE = "Pre-process Database"
TRAIN = "Train Model with Dataset"
RETEST = "Regenerate test predictions (normally done with training)"
EVAL = "Evaluate Trained Model"
main_menu_exits = {PRE: 0, TRAIN: 0, RETEST: 0, EVAL: 0}

selected_tasks = {k: False for k, v in main_menu_exits.items()}
selected_refiners = {k: False for k in ["None", "CorNet", "TreeNet"]}
selected_datasets = set()
selected_models = set()


def add_dataset(dataset):
    global selected_datasets
    try:
        selected_datasets |= {dataset}
    except:
        print(f"Unable to add {dataset}")


def get_datasets():
    global selected_datasets
    return "\n".join(selected_datasets)


def add_model(model):
    global selected_models
    try:
        selected_models |= {model}
    except:
        print(f"Unable to add {model}")


def get_models():
    global selected_models
    return "\n".join(selected_models)


def toggle_task(key):
    global selected_tasks
    selected_tasks[key] = not selected_tasks[key]


def get_tasks():
    global selected_tasks
    return "\n".join([f"{k}: {v}" for k, v in selected_tasks.items()])


def toggle_refiner(key):
    global selected_refiners
    selected_refiners[key] = not selected_refiners[key]


def get_refiners():
    global selected_refiners
    return "\n".join([f"{k}: {v}" for k, v in selected_refiners.items()])


def dataset_selector():
    from global_config import dataset_config_dir

    menu = MultiSelectMenu(
        "Dataset Selection",
        get_datasets,
        epilogue_text=("Please select one or more entries separated by commas, and/or a range "
                       "of numbers. For example:  1,2,3   or   1-4   or   1,3-4"),
        exit_option_text="Back to Main",
    )

    dataset_paths = [str(w) for w in dataset_config_dir.glob("*.yaml")]
    datasets = [(w, safe_load(open(w))["name"]) for w in dataset_paths]

    for path, name in datasets:
        menu.append_item(FunctionItem(name, add_dataset, args=[path]))
    return menu


def model_selector():
    from global_config import model_config_dir

    menu = MultiSelectMenu(
        "Model Selection",
        get_models,
        epilogue_text=("Please select one or more entries separated by commas, and/or a range "
                       "of numbers. For example:  1,2,3   or   1-4   or   1,3-4"),
        exit_option_text="Back to Main",
    )

    model_paths = [str(w) for w in model_config_dir.glob("*.yaml")]
    models = [(w, safe_load(open(w))["name"]) for w in model_paths]

    for path, name in models:
        menu.append_item(FunctionItem(name, add_model, args=[path]))
    return menu


def refiner_selector():
    menu = MultiSelectMenu(
        "Refiner Selection",
        get_refiners,
        epilogue_text=("Please select one or more entries separated by commas, and/or a range "
                       "of numbers. For example:  1,2,3   or   1-4   or   1,3-4"),
        exit_option_text="Back to Main",
    )

    for r in selected_refiners.keys():
        menu.append_item(FunctionItem(r, toggle_refiner, args=[r]))
    return menu


def task_selector():
    menu = MultiSelectMenu(
        "Task Selection",
        get_tasks,
        epilogue_text=("Please select one or more entries separated by commas, and/or a range "
                       "of numbers. For example:  1,2,3   or   1-4   or   1,3-4"),
        exit_option_text="Back to Main",
    )

    for k, v in selected_tasks.items():
        menu.append_item(FunctionItem(k, toggle_task, args=[k]))
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


def retest(dataset_cfg_path, model_cfg_path, refiner):
    pass


def evaluate(dataset_cfg_path, model_cfg_path, refiner):
    from global_config import DatasetConfig
    from evaluation import evaluate

    dataset_params = DatasetConfig(safe_load(open(dataset_cfg_path)))
    model_params = safe_load(open(model_cfg_path))

    evaluate(dataset_params, model_params, refiner)


main_menu_exits = {
    PRE: pre_process,
    TRAIN: train,
    RETEST: retest,
    EVAL: evaluate
}


def all_info():
    return get_tasks() + "\n\n" + get_datasets() + "\n\n" + get_models()


if __name__ == "__main__":
    menu = ConsoleMenu("Main Menu", all_info, exit_option_text="Execute Tasks")
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

    if not any(selected_refiners.values()):
        selected_refiners["None"] = True

    for k, func in main_menu_exits.items():
        if not selected_tasks[k]:
            continue
        for dataset in selected_datasets:
            for refiner in selected_refiners.keys():
                if not selected_refiners[refiner]:
                    continue
                if len(selected_models) < 1:
                    selected_models = {""}
                for model in selected_models:
                    print(f"{k} with {model}-{dataset}-{refiner}")
                    func(dataset, model, refiner)
