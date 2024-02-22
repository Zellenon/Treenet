import warnings

from global_config import DatasetConfig, trained_test_path


import numpy as np
from logzero import logger
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor

from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.evaluation import *
from deepxml.meshprobenet import CorNetMeSHProbeNet, MeSHProbeNet, TreeNetMeSHProbeNet
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
    import joblib

    logger.info("Loading MLB")
    mlb = joblib.load(cfg.label_binarizer)
    all_labels = mlb.classes_
    labels_num = len(all_labels)

    predicted_scores = np.load(str(trained_test_path(cfg,model_cfg,refiner_choice)) 
                               + "-scores.npy", allow_pickle=True)
    predicted_labels_decoded = np.load(str(trained_test_path(cfg,model_cfg,refiner_choice)) 
                                       + "-labels.npy", allow_pickle=True)

    target_labels = np.load(cfg.data["test"].labels_npy, allow_pickle=True)
    logger.info("conversion loop 1 beginning")
    with ThreadPoolExecutor(35) as exec:
        target_score_filter = np.array(
                list(exec.map(lambda row: [w in row for w in all_labels],
                              target_labels)))
    target_scores = np.zeros_like(target_score_filter)
    target_scores[target_score_filter] = 1

    logger.info("Loading labels")
    temp = [
            zip(predicted_labels_decoded[i], predicted_scores[i])
            for i in range(len(predicted_scores))
            ]
    prediction_dicts = [{u: v for u, v in w} for w in temp]
    logger.info("Formatting labels")
    prediction_scores_sorted = [[w.get(v, 0) for v in all_labels]
                                for w in prediction_dicts]
    prediction_scores_sorted = np.array(prediction_scores_sorted)


    logger.info("Running Threshhold Evaluations")
    log = dict()


    threshhold_metrics = dict()
    threshholds = [
            ("HMD", lambda a: np.max(a) / 2),
            ("HMR", lambda a: np.max(a, axis=0) / 2),
            ("ARGHMR", lambda a: np.mean(a[a > np.max(a, axis=0) / 2], axis=0)),
            ("ARGHM", lambda a: np.mean(a[a > np.max(a) / 2], axis=0)),
            ("AGHMR", lambda a: np.mean(a[a > np.max(a, axis=0) / 2])),
            ("AGHM", lambda a: np.mean(a[a > a.max() / 2])),
            ]
    n = [1, 3, 5]
    # averages = ["micro", "macro"]
    averages = ["micro"]
    score_funcs = [
            (metrics.precision_score, "Precision"),
            (metrics.recall_score, "Recall"),
            (metrics.f1_score, "F1"),
            ]

    
    for formula, threshhold in threshholds:
        threshhold_d = threshhold(prediction_scores_sorted)

        pred_filter_d = prediction_scores_sorted > threshhold_d
        pred_d = np.zeros_like(prediction_scores_sorted)
        pred_d[pred_filter_d] = 1

        for average in averages:
            for func, name in score_funcs:
                threshhold_metrics[f"{average}-{name} at p {formula}"] =func(target_scores, pred_d, average=average)
    log["THRESHHOLD METRICS"] = threshhold_metrics

    logger.info("Running TopN Evaluations")

    from deepxml.evaluation import get_f1, get_precision2, get_recall2
    top_n_metrics = dict()
    score_funcs = [
            (get_precision2, "Precision"),
            (get_recall2, "Recall"),
            (get_f1, "F1"),
            ]
    for top_n in n:
        for average in averages:
            for func, name in score_funcs:
                top_n_metrics[f"{average}-{name} at n={top_n}"] = func(predicted_labels_decoded,target_labels, mlb, top=top_n, ave=average)
    log["TOP-N METRICS"] = top_n_metrics

    logger.info("Running DeepXML Evaluations")

    deepxml_metrics = dict()
    score_funcs = [
            (get_p_5, "Precision"),
            (get_n_1), "NDCG1",
            (get_n_3), "NDCG3",
            (get_n_5, "NDCG5"),
            ]
    for func, name in score_funcs:
        deepxml_metrics[f"{name} at n=5"] = func(predicted_labels_decoded,target_labels, mlb)
    log["DeepXML Metrics"] = deepxml_metrics

    logger.info("Running BertMESH Evaluations")

    bertmesh_metrics = dict()
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
    bertmesh_metrics["MiP"] = MiP
    bertmesh_metrics["MiR"] = MiR
    bertmesh_metrics["MiF"] = MiF
    bertmesh_metrics["MaF"] = MaF
    bertmesh_metrics["EBF"] = EBF
    log["BertMESH Metrics"] = bertmesh_metrics  # Hat is predicted

    return log

