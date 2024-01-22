import os

import numpy as np
from logzero import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from db import ChromaLoader
from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.cornet import CorNet, CorNetWrapper
from deepxml.data_utils import get_data, get_mlb, get_word_emb
from deepxml.dataset import MultiLabelDataset
from deepxml.meshprobenet import CorNetMeSHProbeNet, MeSHProbeNet, TreeNetMeSHProbeNet
from deepxml.models import GPipeModel, Model
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from global_config import DatasetConfig, Path, data_dir, results_dir

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


def train_model(cfg: DatasetConfig, model_cnf, refiner):
    model, model_name, data_name = None, model_cnf["name"], cfg.name
    model_path = results_dir / Path("models") / f"{model_name}-{data_name}"
    mlb = get_mlb(
        cfg.label_binarizer,
        None,
    )
    labels_num = len(mlb.classes_)
    emb_init = cfg.emb_init


    logger.info("Loading Training and Validation Set")
    train_x, train_labels = get_data(cfg.data["train"])
    random_state = 1240
    train_x, valid_x, train_labels, valid_labels = train_test_split(
        train_x, train_labels, test_size=cfg.valid_size, random_state=random_state
    )
    train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
    labels_num = len(mlb.classes_)

    logger.info(f"Model Name: {model_name}")
    logger.info(f"Number of Labels: {labels_num}")
    logger.info(f"Size of Training Set: {len(train_x)}")
    logger.info(f"Size of Validation Set: {len(valid_x)}")

    logger.info("Training")
    train_loader = DataLoader(
        MultiLabelDataset(train_x, train_y), cfg.batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        MultiLabelDataset(valid_x, valid_y, training=True),
        cfg.batch_size,
        num_workers=4,
    )
    if "gpipe" not in model_cnf and "gpipe" not in model_cnf["params"]:
        model = Model(
            network=model_dict[model_name],
            labels_num=labels_num,
            model_path=model_path,
            emb_init=emb_init,
            **cfg.model,
            **model_cnf,
        )
    else:
        model = GPipeModel(
            model_name,
            labels_num=labels_num,
            model_path=model_path,
            emb_init=emb_init,
            **cfg.model,
            **model_cnf,
        )
    if refiner == "CorNet":
        model = CorNetWrapper(model)
    model.train(train_loader, valid_loader, **model_cnf["params"])
    # model.train(train_loader, valid_loader)
    logger.info("Finish Training")


