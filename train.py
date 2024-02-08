import numpy as np
from logzero import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from db import ChromaLoader
from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.cornet import CorNet, CorNetWrapper
from deepxml.data_utils import get_data, get_mlb, get_word_emb
from deepxml.dataset import MultiLabelDataset
from deepxml.meshprobenet import CorNetMeSHProbeNet, MeSHProbeNet, TreeNetMeSHProbeNet
from deepxml.models import GPipeModel, Model
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN
from global_config import DatasetConfig, Path, data_dir, results_dir

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


def train_model(cfg: DatasetConfig, model_cnf, refiner):
    model, model_name, data_name = None, model_cnf["name"], cfg.name
    model_path = results_dir / Path("models") / f"{model_name}-{data_name}"
    mlb = get_mlb(
        cfg.label_binarizer,
        None,
    )
    labels_num = len(mlb.classes_)
    emb_init = cfg.emb_init

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
    if "gpipe" not in model_cnf:
        model = Model(
            network=model_dict[model_name],
            labels_num=labels_num,
            model_path=model_path,
            emb_init=emb_init,
            **cfg.model,
            **model_cnf["params"],
        )
    else:
        model = GPipeModel(
            model_name,
            labels_num=labels_num,
            model_path=model_path,
            emb_init=emb_init,
            **cfg.model,
            **model_cnf["params"],
        )
    if refiner == "CorNet":
        model.model = CorNetWrapper(backbone=model.model,
                                    labels_num=labels_num)
    model.train(train_loader, valid_loader, **model_cnf["train"])
    # model.train(train_loader, valid_loader)
    logger.info("Finish Training")

    # Temp code
    logger.info("Loading Test Set")
    import joblib

    mlb = joblib.load(cfg.label_binarizer)
    all_labels = mlb.classes_

    labels_num = len(all_labels)
    logger.info(f"Found a total of {labels_num} labels in the dataset")
    test_x, _ = get_data(cfg.data["test"])
    logger.info(f"Size of Test Set: {len(test_x)}")
    trained_model_path = Path(f"results/models/{model_cfg['name']}-{cfg.name}")

    logger.info("Predicting")
    test_loader = DataLoader(MultiLabelDataset(test_x),
                             cfg.batch_size,
                             num_workers=4)
    model = (GPipeModel(
        model_cfg["name"],
        labels_num=labels_num,
        model_path=trained_model_path,
        emb_init=cfg.emb_init,
        **cfg.model,
        **model_cfg,
    ) if "gpipe" in model_cfg else Model(
        network=model_dict[model_cfg["name"]],
        labels_num=labels_num,
        model_path=trained_model_path,
        emb_init=cfg.emb_init,
        **cfg.model,
        **model_cfg["params"],
    ))
    predicted_scores, predicted_labels_encoded = model.predict(
        test_loader, k=cfg.valid_size)
    predicted_labels_decoded = mlb.classes_[predicted_labels_encoded]
    logger.info("Finish Predicting")
    alllabels = mlb.classes_
    output_res(
        cfg.output_dir,
        f'{model_cfg["name"]}-{cfg.name}',
        predicted_scores,
        predicted_labels_decoded,
    )

    target_labels = np.load(cfg.data["test"].labels_npy, allow_pickle=True)
    target_labels_transformed = mlb.fit_transform(target_labels)
    logger.info("conversion loop 1 beginning")
    target_scores = np.array([[1 if w in row else 0 for w in alllabels]
                              for row in target_labels])

    logger.info("Loading labels")
    temp = [
        zip(predicted_labels_decoded[i], predicted_scores[i])
        for i in range(len(predicted_scores))
    ]
    prediction_dicts = [{u: v for u, v in w} for w in temp]
    print("Formatting labels")
    prediction_scores_sorted = [[w.get(v, 0) for v in alllabels]
                                for w in prediction_dicts]
    prediction_scores_sorted = np.array(prediction_scores_sorted)

    print(target_scores.shape)
    print(prediction_scores_sorted.shape)

    # lower_bound = 0.0
    # threshholds = [lower_bound + (w + 1) * ((1 - lower_bound) / 6) for w in range(0, 5)]
    threshholds = []
    # threshholds = [
    #     np.median(prediction_scores_sorted),
    #     prediction_scores_sorted.mean(),
    #     prediction_scores_sorted.max() / 2,
    # ]
    n = [1, 3, 5]
    averages = ["micro", "macro"]
    score_funcs = [
        (metrics.precision_score, "Precision"),
        (metrics.recall_score, "Recall"),
        (metrics.f1_score, "F1"),
    ]

    for threshhold in threshholds:
        temp_predictions = prediction_scores_sorted > threshhold
        for average in averages:
            for func, name in score_funcs:
                print(
                    f"{average}-{name} at p>{threshhold}: {func(target_scores,temp_predictions, average=average)}"
                )
    from deepxml.evaluation import get_precision2, get_recall2, get_f1

    score_funcs = [
        (get_precision2, "Precision"),
        (get_recall2, "Recall"),
        (get_f1, "F1"),
    ]

    __import__("ipdb").set_trace()
    for top_n in n:
        for average in averages:
            for func, name in score_funcs:
                print(
                    f"{average}-{name} at n={top_n}: {func(predicted_labels_decoded,target_labels, mlb, top=top_n, ave=average)}"
                )
