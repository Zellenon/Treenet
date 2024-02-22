from logzero import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from deepxml.data_utils import get_data, get_mlb
from deepxml.dataset import MultiLabelDataset
from global_config import trained_model_path
from global_config import DatasetConfig, make_model


def train_model(cfg: DatasetConfig, model_cfg, refiner):
    model, model_name = None, model_cfg["name"]
    model_path = trained_model_path(cfg, model_cfg, refiner)
    if model_path.exists():
        model_path.unlink()
    if not model_path.exists():
        print("Successfully deleted trained model")
    else:
        print("Failed to delete trained model")
    mlb = get_mlb(
            cfg.label_binarizer,
            None,
            )
    labels_num = len(mlb.classes_)

    logger.info("Loading Training and Validation Set")
    train_x, train_labels = get_data(cfg.data["train"])
    random_state = 1240
    train_x, valid_x, train_labels, valid_labels = train_test_split(
            train_x,
            train_labels,
            test_size=cfg.valid_size,
            random_state=random_state)
    train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
    labels_num = len(mlb.classes_)

    logger.info(f"Model Name: {model_name}")
    logger.info(f"Number of Labels: {labels_num}")
    logger.info(f"Size of Training Set: {len(train_x)}")
    logger.info(f"Size of Validation Set: {len(valid_x)}")

    logger.info("Training")
    train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
                              cfg.batch_size,
                              shuffle=True,
                              num_workers=4)
    valid_loader = DataLoader(
            MultiLabelDataset(valid_x, valid_y, training=True),
            cfg.batch_size,
            num_workers=4,
            )

    model = make_model(cfg, model_cfg, refiner, labels_num)

    model.train(train_loader, valid_loader, {**model_cfg["train"], **cfg.params})
    logger.info("Finish Training")
