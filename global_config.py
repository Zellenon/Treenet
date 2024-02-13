from pathlib import Path

import numpy as np
from colors import color
from yaml import safe_load

data_dir = Path("data/")
config_dir = Path("configure/")
model_config_dir = config_dir / Path("models")
dataset_config_dir = config_dir / Path("datasets")
results_dir = Path("results/")


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
                           "RETEST" : "Regenerate test predictions",
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

    def exec(self):
        if not any(self.selected_refiners.values()):
            self.selected_refiners["None"] = True

        for k, func in self.tasks.items():
            if not self.selected_tasks[k]:
                continue
            for dataset in self.selected_datasets:
                for refiner in self.selected_refiners.keys():
                    if not self.selected_refiners[refiner]:
                        continue
                    if len(self.selected_models) < 1:
                        self.selected_models = {""}
                    for model in self.selected_models:
                        print(f"{k} with {model}-{dataset}-{refiner}")
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
        return "\n".join([f"{color(k, fg='green' if v else 'red')}" for k, v in self.selected_tasks.items()])
    
    def toggle_refiner(self, key):
        self.selected_refiners[key] = not self.selected_refiners[key]
    
    def get_refiners(self):
        return "  ".join([f"{color(k, fg='green' if v else 'red')}" for k, v in self.selected_refiners.items()])
    
    def all_info(self):
        return "\n\n".join([self.get_refiners(), self.get_tasks(), self.get_datasets(), self.get_models()])
