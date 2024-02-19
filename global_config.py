from pathlib import Path
from deepxml.cornet import CorNetWrapper
from deepxml.models import Model
from deepxml.models_gpipe import GPipeModel
from typing import Dict

import numpy as np
from colors import color
from yaml import safe_load, dump

def ensure(path: Path):
    if not path.exists():
        path.mkdir()

data_dir = Path("data/")
config_dir = Path("configure/")
model_config_dir = config_dir / Path("models")
dataset_config_dir = config_dir / Path("datasets")
results_dir = Path("results/")
result_model_dir = results_dir / Path("models")
result_test_dir = results_dir / Path("test_predictions")
result_log_dir = results_dir / Path("logs")
system_dirs =  [data_dir, config_dir, model_config_dir, dataset_config_dir, results_dir, result_log_dir, result_model_dir, result_test_dir]
for dir in system_dirs:
    ensure(dir)


from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.meshprobenet import CorNetMeSHProbeNet, MeSHProbeNet, TreeNetMeSHProbeNet
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from global_config import Path

model_dict = {
        "AttentionXML": AttentionXML,
        "CorNetAttentionXML": CorNetAttentionXML,
        "MeSHProbeNet": MeSHProbeNet,
        "CorNetMeSHProbeNet": CorNetMeSHProbeNet,
        "TreeNetMeSHProbeNet": TreeNetMeSHProbeNet,
        "BertXML": BertXML,
        "CorNetBertXML": CorNetBertXML,
        "XMLCNN": XMLCNN,
        "CorNetXMLCNN": CorNetXMLCNN,
        }


class DatasetConfig:
    def __init__(self, yaml) -> None:
        self.yaml = yaml
        self.name = yaml["name"]
        self.foldername = yaml.get("foldername", self.name)
        self.folder = data_dir / Path(self.foldername)
        self.data = {
                "train": DatasetSubConfig("train", self.folder, yaml),
                "test": DatasetSubConfig("test", self.folder, yaml),
                }
        self.vocab = self.folder / Path(yaml["vocab"])
        self.emb_init_path = self.folder / Path(yaml["embedding"]["emb_init"])
        try:
            self.emb_init = np.load(self.emb_init_path, allow_pickle=True)
        except:
            pass
        self.label_binarizer = self.folder / Path(yaml.get(*["label_binarizer"] * 2))
        self.valid_size = yaml["valid"]["size"]
        self.batch_size = yaml["batch_size"]
        self.model = yaml["model"]
        self.output_dir = yaml["output"]["res"]


class DatasetSubConfig:
    def __init__(self, split, folder, yaml) -> None:
        self.text = folder / Path(yaml[split]["texts"])
        self.text_npy = self.text.parent / (self.text.name.split(".")[0] + ".npy")
        self.labels = folder / Path(yaml[split]["labels"])
        self.labels_npy = self.labels.parent / (self.labels.name.split(".")[0] + ".npy")

class AppConfig:
    def __init__(self):
        self.task_names = {"PRE" : "Pre-process Database",
                           "TRAIN" : "Train Model with Dataset",
                           "TEST" : "Generate test predictions",
                           "EVAL" : "Evaluate Trained Model"
                           }
        self.tasks = {k:(lambda x,y,z: x+y+z) for k, _ in self.task_names.items()}

        dataset_paths = [str(w) for w in dataset_config_dir.glob("*.yaml")]
        self.all_datasets = {w: safe_load(open(w))["name"] for w in dataset_paths}

        model_paths = [str(w) for w in model_config_dir.glob("*.yaml")]
        self.all_models = {w: safe_load(open(w))["name"] for w in model_paths}

        self.selected_tasks = {k: False for k, _ in self.task_names.items()}
        self.selected_refiners = {k: False for k in ["None", "CorNet", "TreeNet"]}
        self.selected_datasets = set()
        self.selected_models = set()
        self.debug_mode = False

    def save(self):
        data = {
                "selected_datasets": self.selected_datasets,
                "selected_models": self.selected_models,
                "selected_refiners": self.selected_refiners,
                "selected_tasks": self.selected_tasks,
                }
        with open("last_run.yaml", 'w') as f:
            f.write(dump(data))

    def load(self):
        path = Path("last_run.yaml")
        if path.exists():
            with open(path) as f:
                data = safe_load(f.read())
                self.selected_datasets = data["selected_datasets"]
                self.selected_models = data["selected_models"]
                self.selected_refiners = data["selected_refiners"]
                self.selected_tasks = data["selected_tasks"]

    def exec(self):
        self.save()
        if not any(self.selected_refiners.values()):
            self.selected_refiners["None"] = True
        if len(self.selected_models) < 1:
            self.selected_models = {""}

        for dataset in self.selected_datasets:
            for model in self.selected_models:
                for refiner in [k for k, v in self.selected_refiners.items() if v]:
                    for k, func in [(k,v) for k,v in self.tasks.items() if self.selected_tasks[k]]:
                        print(f"{k} with {model}-{dataset}-{refiner}")
                        if not self.debug_mode:
                            try:
                                func(dataset, model, refiner)
                            except Exception as e:
                                print(f"Failed on {k} with {model}-{dataset}-{refiner}")
                                print(e)
                        else:
                            func(dataset, model, refiner)


    def add_dataset(self, dataset):
        self.selected_datasets |= {dataset}

    def get_datasets(self):
        return "\n".join(self.selected_datasets)

    def add_model(self, model):
        self.selected_models |= {model}

    def get_models(self):
        return "\n".join(self.selected_models)

    def toggle_task(self, key):
        self.selected_tasks[key] = not self.selected_tasks[key]

    def get_tasks(self):
        return "\n".join([f"{color(self.task_names[k], fg='green' if v else 'red')}" for k, v in self.selected_tasks.items()])

    def toggle_refiner(self, key):
        self.selected_refiners[key] = not self.selected_refiners[key]

    def get_refiners(self):
        return "  ".join([f"{color(k, fg='green' if v else 'red')}" for k, v in self.selected_refiners.items()])

    def all_info(self):
        return "\n\n\n".join([self.get_refiners(), self.get_tasks(), self.get_datasets(), self.get_models()])



def trained_model_path(cfg: DatasetConfig, model_cfg: Dict, refiner_choice: str):
    return result_model_dir / Path(f"{model_cfg['name']}-{cfg.name}-{refiner_choice}")

def trained_log_path(cfg: DatasetConfig, model_cfg: Dict, refiner_choice: str):
    return result_log_dir / Path(f"{model_cfg['name']}-{cfg.name}-{refiner_choice}")

def trained_test_path(cfg: DatasetConfig, model_cfg: Dict, refiner_choice: str):
    return result_test_dir / Path(f"{model_cfg['name']}-{cfg.name}-{refiner_choice}")

def make_model(cfg: DatasetConfig, model_cfg: Dict, refiner_choice: str, labels_num):
    model = GPipeModel if "gpipe" in model_cfg else Model
    network = model_dict[model_cfg["name"]]
    kwargs = {
            **cfg.model,
            **model_cfg["params"]
            }
    if refiner_choice == "CorNet":
        kwargs["backbone_fn"] = network
        network = CorNetWrapper

    return model(network=network,
                 labels_num=labels_num,
                 model_path=trained_model_path(cfg, model_cfg, refiner_choice),
                 emb_init=cfg.emb_init,
                 **kwargs)
