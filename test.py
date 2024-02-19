from typing import Dict
import warnings
from logzero import logger
from torch.utils.data import DataLoader
from deepxml.data_utils import get_data
from deepxml.dataset import MultiLabelDataset
from global_config import DatasetConfig, make_model
from global_config import trained_test_path
from logzero import logger
from torch.utils.data import DataLoader
from deepxml.data_utils import get_data, output_res
from deepxml.dataset import MultiLabelDataset
from deepxml.evaluation import *

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

    model = make_model(cfg, model_cfg, refiner_choice, labels_num)

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
