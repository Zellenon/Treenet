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

    for k, _ in app_config.selected_tasks.items():
        menu.append_item(FunctionItem(app_config.task_names[k], app_config.toggle_task, args=[k]))
    return menu


def get_settings():
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

    menu.append_item(FunctionItem("Load last run", app_config.load, args=[]))

    menu.start()
    menu.join()
    return app_config

