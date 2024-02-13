from logzero import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.cornet import CorNetWrapper
from deepxml.data_utils import get_data, get_mlb
from deepxml.dataset import MultiLabelDataset
from deepxml.meshprobenet import CorNetMeSHProbeNet, MeSHProbeNet, TreeNetMeSHProbeNet
from deepxml.models import GPipeModel, Model
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from global_config import DatasetConfig, Path,  results_dir

model_dict = {
    "AttentionXML": AttentionXML,
    "CorNetAttentionXML": CorNetAttentionXML,
    "MeSHProbeNet": MeSHProbeNet,
    "CorNetMeSHProbeNet": CorNetMeSHProbeNet,
    "TreeNetMeSHProbeNet": TreeNetMeSHProbeNet,
    "BertXML": BertXML,
    "CorNetBertXML": CorNetBertXML,
    "XMLCNN": XMLCNN,
    "CorNetXMLCNN": CorNetXMLCNN,
}

def delete_old_model(path: Path) -> None:
    if path.exists():
        path.unlink()
    if not path.exists():
        print("Successfully deleted trained model")
    else:
        print("Failed to delete trained model")

def load_training_data(mlb, cfg):
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

    logger.info(f"Number of Labels: {labels_num}")
    logger.info(f"Size of Training Set: {len(train_x)}")
    logger.info(f"Size of Validation Set: {len(valid_x)}")
    return train_x, valid_x, train_y, valid_y

def make_loader(x, y, cfg, **kwargs):
    return DataLoader(MultiLabelDataset(x, y),
                              cfg.batch_size,
                              num_workers=4, **kwargs)

def make_model(model_cfg, cfg, labels_num, model_path) -> GPipeModel | Model:
    model = GPipeModel if "gpipe" in model_cfg else Model
    return model(network=model_dict[model_cfg["name"]],
        labels_num=labels_num,
        model_path=model_path,
        emb_init=cfg.emb_init,
        **cfg.model,
        **model_cfg["params"])

def train_model(cfg: DatasetConfig, model_cfg, refiner):
    model, model_name, data_name = None, model_cfg["name"], cfg.name
    model_path = results_dir / Path("models") / f"{model_name}-{data_name}-{refiner}"
    delete_old_model(model_path)
    logger.info(f"Model Name: {model_name}")

    mlb = get_mlb(
        cfg.label_binarizer,
        None,
    )
    labels_num = len(mlb.classes_)

    train_x, valid_x, train_y, valid_y = load_training_data(mlb, cfg)

    train_loader = make_loader(train_x, train_y, cfg, shuffle=True)
    valid_loader = make_loader(valid_x, valid_y, cfg, training=True)

    model = make_model(model_cfg, cfg, labels_num, model_path)

    if refiner == "CorNet":
        model.model = CorNetWrapper(backbone=model.model,
                                    labels_num=labels_num)

    logger.info("Training")
    model.train(train_loader, valid_loader, **model_cfg["train"])
    # model.train(train_loader, valid_loader)
    logger.info("Finish Training")
