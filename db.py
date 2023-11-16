import chromadb
from more_itertools import batched
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from global_config import DatasetConfig

client = chromadb.PersistentClient(path="db")


def create_collection(name: str):
    client.delete_collection(name=name)
    collection = client.create_collection(name)
    return collection


def get_collection(name: str):
    return client.get_collection(name=name)


def add_database3(cfg):
    try:
        client.delete_collection(cfg.name)
    except:
        pass
    col = client.create_collection(name=cfg.name)
    for split in ["train", "test"]:
        with open(cfg.data[split].text) as text_file:
            with open(cfg.data[split].labels) as label_file:
                batch_size = 50
                text_file = batched(text_file, batch_size)
                label_file = batched(label_file, batch_size)
                for i, lines in tqdm(
                    enumerate(text_file), leave=False, desc="Adding files"
                ):
                    col.add(
                        ids=[
                            str(w)
                            for w in range(i * batch_size, i * batch_size + batch_size)
                        ],
                        documents=lines,
                        metadatas=[{"labels": w} for w in next(label_file)],
                    )


def add_database(cfg: DatasetConfig, replace: bool):
    if not replace:
        # TODO
        print("NOT IMPLEMENTED")
        1 / 0
    try:
        client.delete_collection(cfg.name)
    except:
        pass
    col = client.create_collection(name=cfg.name)
    encoder = SentenceTransformer(
        "sentence-transformers/average_word_embeddings_glove.6B.300d"
    )  # 7 it/s
    # encoder = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2') # 2 it/s
    # encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1') # 2 it/s
    for split in ["train", "test"]:
        with open(cfg.data[split].text) as text_file:
            with open(cfg.data[split].labels) as label_file:
                batch_size = 1000
                text_file = batched(text_file, batch_size)
                label_file = batched(label_file, batch_size)
                for i, lines in tqdm(
                    enumerate(text_file), leave=False, desc="Adding files"
                ):
                    encoded = encoder.encode(lines)
                    col.add(
                        ids=[
                            f"{split}-{w}"
                            for w in range(i * batch_size, i * batch_size + len(lines))
                        ],
                        documents=lines,
                        embeddings=[[v.item() for v in w] for w in encoded],
                        metadatas=[
                            {"labels": ",".join(w.strip().split()), "split": split}
                            for w in next(label_file)
                        ],
                    )


from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

TDataX = Sequence[Sequence]
TDataY = Optional[csr_matrix]


class ChromaLoader(Dataset):
    def __init__(self, cfg: DatasetConfig, mlb, training=True):
        self.col = get_collection(cfg.name)
        self.split = "train" if training else "test"
        self.training = training
        self.mlb = mlb
        self.ids = self.col.get()["ids"]

    def train_ids(self):
        return [w.split("-")[1] for w in self.ids if "train" in w]

    def test_ids(self):
        return [w.split("-")[1] for w in self.ids if "test" in w]

    def __getitem__(self, item):
        if self.training:
            query = self.col.get(
                ids=[f"{self.split}-{item}"], include=["embeddings", "metadatas"]
            )
        else:
            query = self.col.get(ids=[f"{self.split}-{item}"], include=["embeddings"])
        data_x = np.array(query["embeddings"]).squeeze(0).astype(np.float32)
        if self.training:
            data_y = np.array(
                self.mlb.transform(
                    # list(map(lambda x: x.split(","), query["metadatas"]["labels"]))
                    [w["labels"].split(",") for w in query["metadatas"]]
                ).toarray()
            )
            # print("Erroring text:")
            # print(data_y)
            data_y = data_y.squeeze(()).astype(np.float32)
            return data_x, data_y
        else:
            return data_x

    def __len__(self):
        return len(self.ids)
