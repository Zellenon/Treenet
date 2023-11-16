from db import get_collection
from deepxml.data_utils import get_mlb
from global_config import DatasetConfig


def process(params: DatasetConfig):
    print("Deleting existing binarizer if one exists")
    if params.label_binarizer.exists():
        params.label_binarizer.unlink()
    print("Fetching all rows from database")
    query = get_collection(name=params.name).get(include=["metadatas"])
    print("Collecting labels")
    labels = [w["labels"].split(",") for w in query["metadatas"]]
    print("Training MLB on labels")
    get_mlb(params.label_binarizer, labels)
