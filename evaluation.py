import warnings

from global_config import DatasetConfig, results_dir

from pathlib import Path

import numpy as np
from logzero import logger
from sklearn import metrics
from torch.utils.data import DataLoader

from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.data_utils import get_data, get_mlb, output_res
from deepxml.dataset import MultiLabelDataset
from deepxml.evaluation import *
from deepxml.meshprobenet import CorNetMeSHProbeNet, MeSHProbeNet, TreeNetMeSHProbeNet
from deepxml.models import GPipeModel, Model
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN

warnings.filterwarnings("ignore")

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


def evaluate(cfg: DatasetConfig, model_cfg, refiner_choice):
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
    threshholds = [
        np.median(prediction_scores_sorted),
        prediction_scores_sorted.mean(),
        prediction_scores_sorted.max() / 2,
    ]
    n = [1, 3, 5]
    averages = ["micro", "macro"]
    score_funcs = [
        (metrics.precision_score, "Precision"),
        (metrics.recall_score, "Recall"),
        (metrics.f1_score, "F1"),
    ]

    eval_log = open(
        results_dir / Path(model_cfg["name"] + "-" + cfg.name + "-" +
                           refiner_choice + ".txt"),
        "w",
    )

    def log(message):
        print(message)
        eval_log.write(message + "\n")

    log("THRESHHOLD METRICS")
    for threshhold in threshholds:
        temp_predictions = prediction_scores_sorted > threshhold
        for average in averages:
            for func, name in score_funcs:
                log(f"{average}-{name} at p>{threshhold}: {func(target_scores,temp_predictions, average=average)}"
                    )
    from deepxml.evaluation import get_precision2, get_recall2, get_f1

    score_funcs = [
        (get_precision2, "Precision"),
        (get_recall2, "Recall"),
        (get_f1, "F1"),
    ]

    log("TOP-N METRICS")
    # __import__("ipdb").set_trace()
    for top_n in n:
        for average in averages:
            for func, name in score_funcs:
                log(f"{average}-{name} at n={top_n}: {func(predicted_labels_decoded,target_labels, mlb, top=top_n, ave=average)}"
                    )

    log("DeepXML Metrics")
    score_funcs = [
        (get_p_5, "Precision"),
        (get_n_5, "NDCG"),
    ]
    for func, name in score_funcs:
        log(f"{name} at n=5: {func(predicted_labels_decoded,target_labels, mlb)}"
            )

    log("BertMESH Metrics")  # Hat is predicted
    MiP = np.sum(predicted_scores * target_scores) / np.sum(predicted_scores)
    MiR = np.sum(predicted_scores * target_scores) / np.sum(target_scores)
    MiF = (2 * MiP * MiR) / (MiP + MiR)
    MaPx = np.sum(predicted_scores * target_scores, axis=1) / np.sum(
        predicted_scores, axis=1)
    MaRx = np.sum(predicted_scores * target_scores, axis=1) / np.sum(
        target_scores, axis=1)
    MaF = (1 / labels_num) * np.sum((2 * MaPx * MaRx) / (MaPx + MaRx))
    EBPi = np.sum(predicted_scores * target_scores, axis=0) / np.sum(
        predicted_scores, axis=0)
    EBRi = np.sum(predicted_scores * target_scores, axis=0) / np.sum(
        target_scores, axis=0)
    EBF = (1 / EBRi.shape[0]) * np.sum((2 * EBPi * EBRi) / (EBPi + EBRi))
    log(f"MiP: {MiP}; MiR: {MiR}; MiF: {MiF}")
    log(f"MaF: {MaF}; EBF: {EBF}")


if __name__ == "__main__":
    main()
