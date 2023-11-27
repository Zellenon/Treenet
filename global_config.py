from pathlib import Path

import numpy as np
# import gensim.downloader

data_dir = Path("data/")
config_dir = Path("configure/")
model_config_dir = config_dir / Path("models")
dataset_config_dir = config_dir / Path("datasets")
results_dir = Path("results/")
# w2v = gensim.downloader.load("conceptnet-numberbatch-17-06-300")


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
        self.emb_init = np.load(self.emb_init_path, allow_pickle=True)
        self.label_binarizer = self.folder / Path(yaml.get(*["label_binarizer"] * 2))
        self.valid_size = yaml["valid"]["size"]
        self.batch_size = yaml["batch_size"]
        self.model = yaml["model"]


class DatasetSubConfig:
    def __init__(self, split, folder, yaml) -> None:
        self.text = folder / Path(yaml[split]["texts"])
        self.text_npy = self.text.parent / (self.text.name + ".npy")
        self.labels = folder / Path(yaml[split]["labels"])
        self.labels_npy = self.labels.parent / (self.labels.name + ".npy")
