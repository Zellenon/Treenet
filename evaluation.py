import warnings

from global_config import DatasetConfig

warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from logzero import logger
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.data_utils import get_data, get_mlb, output_res
from deepxml.dataset import MultiLabelDataset
from deepxml.evaluation import *
from deepxml.meshprobenet import CorNetMeSHProbeNet, MeSHProbeNet, TreeNetMeSHProbeNet
from deepxml.models import GPipeModel, Model
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN

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


def evaluate(cfg: DatasetConfig, model_cfg):
    logger.info("Loading Test Set")
    # mlb = get_mlb(cfg.label_binarizer)
    import joblib

    mlb = joblib.load(cfg.label_binarizer)
    # ipdb.set_trace()
    all_labels = mlb.classes_

    labels_num = len(all_labels)
    print(f"Found a total of {labels_num} labels in the dataset")
    test_x, _ = get_data(cfg.data["test"])
    logger.info(f"Size of Test Set: {len(test_x)}")
    trained_model_path = Path(f"results/models/{model_cfg['name']}-{cfg.name}")

    logger.info("Predicting")
    test_loader = DataLoader(MultiLabelDataset(test_x), cfg.batch_size, num_workers=4)
    model = (
        GPipeModel(
            model_cfg["name"],
            labels_num=labels_num,
            model_path=trained_model_path,
            emb_init=cfg.emb_init,
            **cfg.model,
            **model_cfg,
        )
        if "gpipe" in model_cfg
        else Model(
            network=model_dict[model_cfg["name"]],
            labels_num=labels_num,
            model_path=trained_model_path,
            emb_init=cfg.emb_init,
            **cfg.model,
            **model_cfg,
        )
    )
    scores, labels = model.predict(test_loader, k=cfg.valid_size)
    logger.info("Finish Predicting")
    labels = mlb.classes_[labels]
    output_res(cfg.output_dir, f'{model_cfg["name"]}-{cfg.name}', scores, labels)

    target_labels = np.load(targets, allow_pickle=True)
    print("conversion loop 1 beginning")
    targets = np.array(
        [[1 if w in row else 0 for w in alllabels] for row in target_labels]
    )

    print("Loading labels")
    predicted_labels = np.load(labels, allow_pickle=True)
    predicted_scores = np.load(scores, allow_pickle=True)
    temp = [
        zip(predicted_labels[i], predicted_scores[i])
        for i in range(len(predicted_scores))
    ]
    prediction_dicts = [{u: v for u, v in w} for w in temp]
    print("Formatting labels")
    predictions = [[w.get(v, 0) for v in alllabels] for w in prediction_dicts]
    predictions = np.array(predictions)

    print(targets.shape)
    print(predictions.shape)

    # threshholds = [0.8 + w/110 for w in range(0,20)]
    threshholds = [h + (w + 1) * ((1 - h) / 21) for w in range(0, 20)]
    n = [1, 3, 5]
    averages = ["micro", "macro"]
    score_funcs = [
        (metrics.precision_score, "Precision"),
        (metrics.recall_score, "Recall"),
        (metrics.f1_score, "F1"),
    ]
    # score_funcs = [(metrics.precision_score, "Precision")]

    for threshhold in threshholds:
        temp_predictions = predictions > threshhold
        for average in averages:
            for func, name in score_funcs:
                print(
                    f"{average}-{name} at p>{threshhold}: {func(targets,temp_predictions, average=average)}"
                )
    for top_n in n:
        temp_predictions = np.zeros_like(predictions)
        indices = np.argpartition(predictions, -1)[-1 * top_n :]
        temp_predictions[indices] = 1

        for average in averages:
            for func, name in score_funcs:
                print(
                    f"{average}-{name} at n={top_n}: {func(targets,temp_predictions, average=average)}"
                )


# @click.command()
# @click.option('-r', '--results', type=click.Path(exists=True), help='Path of results.')
# @click.option('-t', '--targets', type=click.Path(exists=True), help='Path of targets.')
# @click.option('--train-labels', type=click.Path(exists=True), default=None, help='Path of labels for training set.')
# @click.option('-a', type=click.FLOAT, default=0.55, help='Parameter A for propensity score.')
# @click.option('-b', type=click.FLOAT, default=1.5, help='Parameter B for propensity score.')
# def main(results, targets, train_labels, a, b):
#     res, targets = np.load(results, allow_pickle=True), np.load(targets, allow_pickle=True)
#     mlb = MultiLabelBinarizer(sparse_output=True)
#     targets = mlb.fit_transform(targets)
#     # print('Precision@1,3,5:', get_p_1(res, targets, mlb), get_p_3(res, targets, mlb), get_p_5(res, targets, mlb))
#     print('MicroPrecision@1,3,5:', get_p_1(res, targets, mlb, ave='micro'), get_p_3(res, targets, mlb, ave='micro'), get_p_5(res, targets, mlb, ave='micro'))
#     print('MacroPrecision@1,3,5:', get_p_1(res, targets, mlb, ave='macro'), get_p_3(res, targets, mlb, ave='macro'), get_p_5(res, targets, mlb, ave='macro'))
#     print('MicroRecall@1,3,5:', get_r_1(res, targets, mlb, ave='micro'), get_r_3(res, targets, mlb, ave='micro'), get_r_5(res, targets, mlb, ave='micro'))
#     print('MacroRecall@1,3,5:', get_r_1(res, targets, mlb, ave='macro'), get_r_3(res, targets, mlb, ave='macro'), get_r_5(res, targets, mlb, ave='macro'))
#     print('MicroF1@1,3,5:', get_f_1(res, targets, mlb, ave='micro'), get_f_3(res, targets, mlb, ave='micro'), get_f_5(res, targets, mlb, ave='micro'))
#     print('MacroF1@1,3,5:', get_f_1(res, targets, mlb, ave='macro'), get_f_3(res, targets, mlb, ave='macro'), get_f_5(res, targets, mlb, ave='macro'))
#     # print('MicroPrecision@1,3,5', micro_precision(1), micro_precision(3), micro_precision(5))
#     print('nDCG@1,3,5:', get_n_1(res, targets, mlb), get_n_3(res, targets, mlb), get_n_5(res, targets, mlb))
#     if train_labels is not None:
#         train_labels = np.load(train_labels, allow_pickle=True)
#         inv_w = get_inv_propensity(mlb.transform(train_labels), a, b)
#         print('PSPrecision@1,3,5:', get_psp_1(res, targets, inv_w, mlb), get_psp_3(res, targets, inv_w, mlb),
#               get_psp_5(res, targets, inv_w, mlb))
#         print('PSnDCG@1,3,5:', get_psndcg_1(res, targets, inv_w, mlb), get_psndcg_3(res, targets, inv_w, mlb),
#               get_psndcg_5(res, targets, inv_w, mlb))
#
#
if __name__ == "__main__":
    main()
