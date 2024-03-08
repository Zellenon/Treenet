# from db import get_collection
import re

spliter = re.compile(r"[ ,\t\n]")
from functools import reduce

import numpy as np
from logzero import logger
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from deepxml.data_utils import build_vocab, convert_to_binary, get_mlb
from global_config import DatasetConfig


def tokenize_line(sentence: str):
    # We added a /SEP/ symbol between titles and descriptions such as Amazon datasets.
    return [
        token.lower() if token != "/SEP/" else token
        for token in word_tokenize(sentence)
        if len(re.sub(r"[^\w]", "", token)) > 0
    ]


def tokenize_file(file):
    with ThreadPoolExecutor(35) as exec:
        return list(tqdm(exec.map(tokenize_line, file), desc="Tokenizing"))


def process(cfg: DatasetConfig, vocab_size: int, max_len: int):
    for split in ["train", "test"]:
        textpath = cfg.data[split].text
        outpath = cfg.data[split].text_npy
        with open(textpath) as text_file, open(outpath, "w") as output_file:
            for line in tokenize_file(text_file):
                print(*line, file=output_file)

    vocab = process_vocab(cfg, vocab_size)

    build_binarizer(cfg)

    for split in ["train", "test"]:
        textpath = cfg.data[split].text
        labelpath = cfg.data[split].labels
        logger.info(f"Getting Dataset: {textpath} Max Length: {labelpath}")
        texts, labels = convert_to_binary(
            textpath.parent / (textpath.name + ""), labelpath, max_len, vocab
        )
        [[n for n in w if len(n) > 0] for w in labels]
        logger.info(f"Size of Samples: {len(texts)}")
        np.save(cfg.data[split].text_npy, texts)
        if labels is not None:
            assert len(texts) == len(labels)
            np.save(cfg.data[split].labels_npy, labels)


def process_vocab(cfg: DatasetConfig, vocab_size: int):
    logger.info(f"Building Vocab.")
    from w2vfile import w2v

    logger.info(f"Imported w2v model")
    with open(cfg.data["train"].text) as fp:
        vocab, emb_init = build_vocab(fp, w2v, vocab_size=vocab_size)
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

    print(labels)
    print("Training MLB on labels")
    get_mlb(cfg.label_binarizer, [labels])
