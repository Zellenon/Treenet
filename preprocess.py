from db import get_collection
from deepxml.data_utils import get_mlb
from global_config import DatasetConfig, w2v
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

def process(params: DatasetConfig, vocab_size: int, max_len: int):
    for split in ["train", "test"]:
        textpath = params.data[split].text
        with open(textpath) as text_file, open(textpath.parent / (textpath.name + '.npy'), 'w') as output_file:
            for line in tqdm(text_file, desc='Tokenizing'):
                print(*tokenize(line), file=output_file)

    logger.info(F'Building Vocab.')
    with open(params.data["train"].text) as fp:
        vocab, emb_init = build_vocab(fp, w2v, vocab_size=vocab_size)
    np.save(params.vocab, vocab)
    np.save(params.emb_init_path, emb_init)
    vocab = {word: i for i, word in enumerate(np.load(params.vocab))}
    logger.info(F'Vocab Size: {len(vocab)}')

    print("Deleting existing binarizer if one exists")
    if params.label_binarizer.exists():
        params.label_binarizer.unlink()
    print("Fetching all rows from database")
    query = get_collection(name=params.name).get(include=["metadatas"])
    print("Collecting labels")
    labels = [w["labels"].split(",") for w in query["metadatas"]]
    print("Training MLB on labels")
    get_mlb(params.label_binarizer, labels)

    for split in ["train", "test"]:
        textpath = params.data[split].text
        labelpath = params.data[split].labels
        logger.info(F'Getting Dataset: {textpath} Max Length: {labelpath}')
        texts, labels = convert_to_binary(textpath.parent / (textpath.name + ""), labelpath, max_len, vocab)
        logger.info(F'Size of Samples: {len(texts)}')
        np.save(textpath.parent / (textpath.name + '.npy'), texts)
        if labels is not None:
            assert len(texts) == len(labels)
            np.save(labelpath.parent / (labelpath.name + '.npy'), labels)
