# from db import get_collection
from deepxml.data_utils import get_mlb
from global_config import DatasetConfig
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re
from logzero import logger
from deepxml.data_utils import build_vocab, convert_to_binary
import numpy as np

def tokenize(sentence: str, sep='/SEP/'):
    # We added a /SEP/ symbol between titles and descriptions such as Amazon datasets.
    return [token.lower() if token != sep else token for token in word_tokenize(sentence)
            if len(re.sub(r'[^\w]', '', token)) > 0]

def process(cfg: DatasetConfig, vocab_size: int, max_len: int):
    for split in ["train", "test"]:
        textpath = cfg.data[split].text
        with open(textpath) as text_file, open(textpath.parent / (textpath.name + '.npy'), 'w') as output_file:
            for line in tqdm(text_file, desc='Tokenizing'):
                print(*tokenize(line), file=output_file)

    vocab = process_vocab(cfg, vocab_size)

    build_binarizer(cfg)

    for split in ["train", "test"]:
        textpath = cfg.data[split].text
        labelpath = cfg.data[split].labels
        logger.info(F'Getting Dataset: {textpath} Max Length: {labelpath}')
        texts, labels = convert_to_binary(textpath.parent / (textpath.name + ""), labelpath, max_len, vocab)
        logger.info(F'Size of Samples: {len(texts)}')
        np.save(textpath.parent / (textpath.name + '.npy'), texts)
        if labels is not None:
            assert len(texts) == len(labels)
            np.save(labelpath.parent / (labelpath.name + '.npy'), labels)

def process_vocab(cfg: DatasetConfig, vocab_size: int):
    logger.info(F'Building Vocab.')
    from w2vfile import w2v
    logger.info(F'Imported w2v model')
    with open(cfg.data["train"].text) as fp:
        vocab, emb_init = build_vocab(fp, w2v, vocab_size=vocab_size)
    np.save(cfg.vocab, vocab)
    np.save(cfg.emb_init_path, emb_init)
    vocab = {word: i for i, word in enumerate(np.load(cfg.vocab))}
    logger.info(F'Vocab Size: {len(vocab)}')
    return vocab

def build_binarizer(cfg: DatasetConfig):
    print("Deleting existing binarizer if one exists")
    if cfg.label_binarizer.exists():
        cfg.label_binarizer.unlink()
    print("Collecting labels")
    labels = [set(line.split()) for line in open(cfg.data["train"].labels).readlines()]
    labels = map(lambda x: sum(x, {}), labels)
    labels = sum(labels)
    print("Training MLB on labels")
    get_mlb(cfg.label_binarizer, labels)
