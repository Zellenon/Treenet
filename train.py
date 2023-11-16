import os

import numpy as np
from logzero import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from db import ChromaLoader
from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.data_utils import get_data, get_mlb, get_word_emb
from deepxml.dataset import MultiLabelDataset
from deepxml.meshprobenet import CorNetMeSHProbeNet, MeSHProbeNet, TreeNetMeSHProbeNet
from deepxml.models import GPipeModel, Model
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from global_config import Path, data_dir, results_dir

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


def train_model(cfg, model_cnf, refiner):
    model, model_name, data_name = None, model_cnf["name"], cfg.name
    model_path = results_dir / Path("models") / f"{model_name}-{data_name}"

    mlb = get_mlb(
        cfg.label_binarizer,
        None,
    )
    labels_num = len(mlb.classes_)

    train_set = ChromaLoader(cfg, mlb)
    train_ids = train_set.train_ids()
    np.random.shuffle(train_set.train_ids())
    # TODO: Add support for configured valid splits
    valid_count = int(len(train_ids) * 0.2)
    train_x, valid_x = train_ids[:-valid_count], train_ids[valid_count:]

    train_loader = DataLoader(
        train_set,
        cfg.batch_size,
        shuffle=True,
        num_workers=4,
    )
    valid_loader = DataLoader(
        ChromaLoader(cfg, mlb, training=False),
        cfg.batch_size,
        num_workers=4,
    )
    logger.info(f"Model Name: {model_name}")

    logger.info("Loading Training and Validation Set")
    # random_state = data_cnf["valid"].get("random_state", 1240)

    labels_num = len(mlb.classes_)
    logger.info(f"Number of Labels: {labels_num}")
    logger.info(f"Size of Training Set: {len(train_x)}")
    logger.info(f"Size of Validation Set: {len(valid_x)}")

    logger.info("Training")
    # train_loader = DataLoader(
    #     MultiLabelDataset(train_x, train_y),
    #     data_cnf["train"]["batch_size"],
    #     shuffle=True,
    #     num_workers=4,
    # )
    if "gpipe" not in model_cnf:
        model = Model(
            network=model_dict[model_name],
            labels_num=len(mlb.classes_),
            model_path=model_path,
            emb_init=cfg.emb_init,
            **cfg.model,
            **model_cnf["model"],
        )
    else:
        model = GPipeModel(
            model_name,
            labels_num=labels_num,
            model_path=model_path,
            emb_init=cfg.emb_init,
            # **data_cnf["model"],
            **model_cnf["model"],
        )
    # model.train(train_loader, valid_loader, **data_cnf["train"])
    model.train(train_loader, valid_loader)
    logger.info("Finish Training")
