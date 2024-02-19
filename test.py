from typing import Dict
import warnings
from logzero import logger
from torch.utils.data import DataLoader
from deepxml.cornet import CorNetWrapper
from deepxml.data_utils import get_data
from deepxml.dataset import MultiLabelDataset
from deepxml.models import Model
from deepxml.models_gpipe import GPipeModel
from global_config import DatasetConfig, Path
from pathlib import Path
from logzero import logger
from torch.utils.data import DataLoader
from deepxml.data_utils import get_data, output_res
from deepxml.dataset import MultiLabelDataset
from deepxml.evaluation import *
from deepxml.models import Model
from deepxml.models_gpipe import GPipeModel

from global_config import model_dict, trained_model_path, trained_test_path

warnings.filterwarnings("ignore")

def predict(cfg: DatasetConfig, model_cfg: Dict, refiner_choice: str):
    import joblib

    logger.info("Loading MLB")
    mlb = joblib.load(cfg.label_binarizer)
    all_labels = mlb.classes_
    labels_num = len(all_labels)
    test_x, _ = get_data(cfg.data["test"])

    logger.info(f"Found a total of {labels_num} labels in the dataset")
    logger.info(f"Size of Test Set: {len(test_x)}")
    logger.info("Predicting")

    test_loader = DataLoader(MultiLabelDataset(test_x),
                             cfg.batch_size,
                             num_workers=4)

    model = GPipeModel if "gpipe" in model_cfg else Model
    network = model_dict[model_cfg["name"]]
    kwargs = {
            **cfg.model,
            **model_cfg["params"]
            }
    if refiner_choice == "CorNet":
        kwargs["backbone_fn"] = network
        network = CorNetWrapper

    model = model(network=network,
                  labels_num=labels_num,
                  model_path=trained_model_path(cfg, model_cfg, refiner_choice),
                  emb_init=cfg.emb_init,
                  **kwargs)

    predicted_scores, predicted_labels_encoded = model.predict(
            test_loader, k=labels_num) # TODO: This used to be cfg.valid_size. Maybe we need to change?
    predicted_labels_decoded = mlb.classes_[predicted_labels_encoded]
    logger.info("Finish Predicting")

    output_res(
            trained_test_path(cfg, model_cfg, refiner_choice),
            f'{model_cfg["name"]}-{cfg.name}-{refiner_choice}',
            predicted_scores,
            predicted_labels_decoded,
            )
