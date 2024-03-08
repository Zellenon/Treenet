# from db import get_collection
import re

spliter = re.compile(r"[ ,\t\n]")
from functools import reduce

import numpy as np
from logzero import logger
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from deepxml.data_utils import build_vocab, get_mlb, text_to_binary, labels_to_binary
from global_config import DatasetConfig


def tokenize_line(sentence: str):
    # We added a /SEP/ symbol between titles and descriptions such as Amazon datasets.
    return [
        token.lower() if token != "/SEP/" else token
        for token in word_tokenize(sentence)
        if len(re.sub(r"[^\w]", "", token)) > 0
    ]


def tokenize_file(file):
    return process_map(tokenize_line, file, desc="Tokenizing", max_workers=50, chunksize=80)


def process(cfg: DatasetConfig):
    logger.info(f"Tokenizing.")
    for split in ["train", "test"]:
        textpath = cfg.data[split].text
        outpath = cfg.data[split].text_npy
        with open(textpath) as text_file, open(outpath, "w") as output_file:
            for line in tqdm(tokenize_file(text_file), desc="Outputting tokens"):
                print(*line, file=output_file)

    logger.info(f"Building Vocab.")
    vocab = process_vocab(cfg)

    logger.info(f"Building binarizer.")
    build_binarizer(cfg)

    for split in ["train", "test"]:
        textpath = cfg.data[split].text
        labelpath = cfg.data[split].labels
        logger.info(f"Getting Dataset: {textpath} Max Length: {labelpath}")

        texts = text_to_binary(textpath.parent / (textpath.name + ""), cfg.text_len, vocab)
        labels = labels_to_binary(labelpath)

        [[n for n in w if len(n) > 0] for w in labels]
        logger.info(f"Size of Samples: {len(texts)}")
        np.save(cfg.data[split].text_npy, texts)
        if labels is not None:
            assert len(texts) == len(labels)
            np.save(cfg.data[split].labels_npy, labels)


def process_vocab(cfg: DatasetConfig):
    from w2vfile import w2v

    logger.info(f"Imported w2v model")
    with open(cfg.data["train"].text) as fp:
        vocab, emb_init = build_vocab(fp, w2v, vocab_size=cfg.vocab_size)
    np.save(cfg.vocab, vocab)
    np.save(cfg.emb_init_path, emb_init)
    vocab = {word: i for i, word in enumerate(np.load(cfg.vocab))}
    logger.info(f"Vocab Size: {len(vocab)}")
    return vocab


def build_binarizer(cfg: DatasetConfig):
    print("Deleting existing binarizer if one exists")
    if cfg.label_binarizer.exists():
        cfg.label_binarizer.unlink()
    print("Collecting labels")

    def reduce_fun(a, b):
        return a | b

    labels = reduce(
        reduce_fun,
        [
            set(re.split(spliter, line))
            for line in open(cfg.data["train"].labels).readlines()
        ],
    )

    labels = labels | reduce(
        reduce_fun,
        [
            set(re.split(spliter, line))
            for line in open(cfg.data["test"].labels).readlines()
        ],
    )

    labels = {w for w in labels if len(w) > 0}

    print("Training MLB on labels")
    get_mlb(cfg.label_binarizer, [labels])
