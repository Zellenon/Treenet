
from consolemenu import ConsoleMenu, MultiSelectMenu
from consolemenu.items import SubmenuItem
from consolemenu.items import FunctionItem
from yaml import safe_load
import menu
from global_config import AppConfig, result_log_dir

def pre_process(dataset_cfg_path, _, _2):
    from global_config import DatasetConfig
    from preprocess import process
    cfg = DatasetConfig(safe_load(open(dataset_cfg_path)))
    process(cfg)


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
                  result_log_dir / (model_params["name"] + "-" + dataset_params.name + "-" +
                                     refiner + ".txt"), 'w'
                                     ) as logfile:
        logfile.write("\n\n".join([f"{secname}:\n" +"\n".join(
            [f"{metric}: {score}" for metric,score in sec.items()]
            ) for secname, sec in results.items()]))



menu.app_config.tasks = {
    "PRE": pre_process,
    "TRAIN": train,
    "TEST": test,
    "EVAL": evaluate
}

if __name__ == "__main__":
    settings = menu.get_settings()
    settings.exec()
