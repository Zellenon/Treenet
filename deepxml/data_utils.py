import os
import re
from collections import Counter
from typing import Iterable, Union

import joblib
import numpy as np
from gensim.models import KeyedVectors
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from tqdm import tqdm

spliter = re.compile(r"[ ,\t\n]")
from typing import Dict, Iterable, Tuple, Union

from global_config import DatasetSubConfig


def build_vocab(
    texts: Iterable,
    w2v_model,
    vocab_size=500000,
    pad="<PAD>",
    unknown="<UNK>",
    sep="/SEP/",
    max_times=1,
    freq_times=1,
):
    # w2v_model = KeyedVectors.load(str(w2v_model))
    emb_size = w2v_model.vector_size
    vocab, emb_init = (
        [pad, unknown],
        [
            np.zeros(emb_size),
            np.random.uniform(-1.0, 1.0, emb_size),
        ],
    )
    counter = Counter(token for t in texts for token in set(t.split()))
    for word, freq in sorted(
        counter.items(), key=lambda x: (x[1], x[0] in w2v_model), reverse=True
    ):
        if word in w2v_model or freq >= freq_times:
            vocab.append(word)
            # We used embedding of '.' as embedding of '/SEP/' symbol.
            word = "." if word == sep else word
            emb_init.append(
                w2v_model[word]
                if word in w2v_model
                else np.random.uniform(-1.0, 1.0, emb_size)
            )
        if freq < max_times or vocab_size == len(vocab):
            break
    return np.asarray(vocab), np.asarray(emb_init)


def get_word_emb(vec_path, vocab_path=None):
    if vocab_path is not None:
        with open(vocab_path) as fp:
            vocab = {word: idx for idx, word in enumerate(fp)}
        return np.load(vec_path), vocab
    else:
        return np.load(vec_path)


def get_data(cfg_data: DatasetSubConfig):
    return (
        np.load(cfg_data.text_npy, allow_pickle=True),
        np.load(cfg_data.labels_npy, allow_pickle=True)
        if cfg_data.labels_npy is not None
        else None,
    )


def convert_to_binary(
    text_file, label_file=None, max_len=None, vocab=None, pad="<PAD>", unknown="<UNK>"
):
    with open(text_file) as fp:
        texts = [
            [vocab.get(word, vocab[unknown]) for word in line.split()]
            for line in tqdm(fp, desc="Converting token to id", leave=False)
        ]
        # print([len(w) for w in texts])
        # texts = np.asarray(texts)
    labels = None
    if label_file is not None:
        with open(label_file) as fp:
            labels = np.array(
                [
                    [label for label in re.split(spliter, line)]
                    for line in tqdm(fp, desc="Converting labels", leave=False)
                ],
                dtype=object,
            )
    return truncate_text(texts, max_len, vocab[pad], vocab[unknown]), labels


def truncate_text(texts, max_len=500, padding_idx=0, unknown_idx=1):
    if max_len is None:
        return texts
    texts = np.asarray(
        [list(x[:max_len]) + [padding_idx] * (max_len - len(x)) for x in texts]
    )
    texts[(texts == padding_idx).all(axis=1), 0] = unknown_idx
    return texts


def get_mlb(mlb_path, labels=None) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def get_sparse_feature(feature_file, label_file):
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    return normalize(sparse_x), np.load(label_file) if label_file is not None else None


def output_res(output_path, name, scores, labels):
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, f"{name}-scores"), scores)
    np.save(os.path.join(output_path, f"{name}-labels"), labels)
