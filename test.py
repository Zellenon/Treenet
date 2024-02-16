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

from global_config import model_dict, result_test_dir

warnings.filterwarnings("ignore")

def predict(cfg: DatasetConfig, model_cfg, refiner_choice):
    import joblib

    logger.info("Loading MLB")
    mlb = joblib.load(cfg.label_binarizer)
    all_labels = mlb.classes_
    labels_num = len(all_labels)
    test_x, _ = get_data(cfg.data["test"])
    trained_model_path = Path(f"results/models/{model_cfg['name']}-{cfg.name}")

    logger.info(f"Found a total of {labels_num} labels in the dataset")
    logger.info(f"Size of Test Set: {len(test_x)}")
    logger.info("Predicting")

    test_loader = DataLoader(MultiLabelDataset(test_x),
                             cfg.batch_size,
                             num_workers=4)
    model = GPipeModel if "gpipe" in model_cfg else Model
    model = model(network=model_dict[model_cfg["name"]],
                  labels_num=labels_num,
                  model_path=trained_model_path,
                  emb_init=cfg.emb_init,
                  **cfg.model,
                  **model_cfg["params"])
    if refiner_choice == "CorNet":
        model.model = CorNetWrapper(backbone=model.model,
                                    labels_num=labels_num)

    predicted_scores, predicted_labels_encoded = model.predict(
            test_loader, k=labels_num) # TODO: This used to be cfg.valid_size. Maybe we need to change?
    predicted_labels_decoded = mlb.classes_[predicted_labels_encoded]
    logger.info("Finish Predicting")

    output_res(
            result_test_dir,
            f'{model_cfg["name"]}-{cfg.name}-{refiner_choice}',
            predicted_scores,
            predicted_labels_decoded,
            )
