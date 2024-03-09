from functools import partial
from typing import Hashable, Iterable, List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

TPredict = np.ndarray
TTarget = Union[Iterable[Iterable[Hashable]], csr_matrix]
TMlb = Optional[MultiLabelBinarizer]
TClass = Optional[List[Hashable]]


def get_mlb(classes: TClass = None, mlb: TMlb = None, targets: TTarget = None):
    # ipdb.set_trace()
    if classes is not None:
        mlb = MultiLabelBinarizer(sparse_output=True)
        # mlb = MultiLabelBinarizer(classes, sparse_output=True)
    if mlb is None and targets is not None:
        if isinstance(targets, csr_matrix):
            # mlb = MultiLabelBinarizer(range(targets.shape[1]), sparse_output=True)
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit([range(targets.shape[1])])
        else:
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit(targets)
    return mlb


def get_precision(
    prediction: TPredict,
    targets: TTarget,
    mlb: TMlb = None,
    classes: TClass = None,
    top=5,
):
    mlb = get_mlb(classes, mlb, targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top])
    return prediction.multiply(targets).sum() / (top * targets.shape[0])


def get_precision2(
    prediction: TPredict,
    targets: TTarget,
    mlb: TMlb = None,
    classes: TClass = None,
    top=5,
    ave="binary",
):
    if ave == "binary":
        return get_precision(prediction, targets, mlb, classes, top)
    mlb = get_mlb(classes, mlb, targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top])
    # return prediction.multiply(targets).sum() / (top * targets.shape[0])
    return metrics.precision_score(targets, prediction, average=ave)


get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)
get_p_7 = partial(get_precision, top=7)
get_p_9 = partial(get_precision, top=9)
get_p_11 = partial(get_precision, top=11)
get_p_10 = partial(get_precision, top=10)


def get_recall2(
    prediction: TPredict,
    targets: TTarget,
    mlb: TMlb = None,
    classes: TClass = None,
    top=5,
    ave="micro",
):
    # if ave == 'binary':
    #     return get_precision(prediction, targets, mlb, classes, top)
    mlb = get_mlb(classes, mlb, targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top])
    # return prediction.multiply(targets).sum() / (top * targets.shape[0])
    return metrics.recall_score(targets, prediction, average=ave)


get_r_1 = partial(get_recall2, top=1)
get_r_3 = partial(get_recall2, top=3)
get_r_5 = partial(get_recall2, top=5)
get_r_7 = partial(get_recall2, top=7)
get_r_9 = partial(get_recall2, top=9)
get_r_11 = partial(get_recall2, top=11)
get_r_10 = partial(get_recall2, top=10)


def get_f1(
    prediction: TPredict,
    targets: TTarget,
    mlb: TMlb = None,
    classes: TClass = None,
    top=5,
    ave="binary",
):
    # if ave == 'binary':
    #     return get_precision(prediction, targets, mlb, classes, top)
    mlb = get_mlb(classes, mlb, targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top])
    # return prediction.multiply(targets).sum() / (top * targets.shape[0])
    return metrics.f1_score(targets, prediction, average=ave)


get_f_1 = partial(get_f1, top=1)
get_f_3 = partial(get_f1, top=3)
get_f_5 = partial(get_f1, top=5)
get_f_7 = partial(get_f1, top=7)
get_f_9 = partial(get_f1, top=9)
get_f_11 = partial(get_f1, top=11)
get_f_10 = partial(get_f1, top=10)


def get_ndcg(
    prediction: TPredict,
    targets: TTarget,
    mlb: TMlb = None,
    classes: TClass = None,
    top=5,
):
    mlb = get_mlb(classes, mlb, targets)
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((targets.shape[0], 1))
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    for i in range(top):
        p = mlb.transform(prediction[:, i : i + 1])
        dcg += p.multiply(targets).sum(axis=-1) * log[i]
    return np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top) - 1])


get_n_1 = partial(get_ndcg, top=1)
get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)
get_n_7 = partial(get_ndcg, top=7)
get_n_9 = partial(get_ndcg, top=9)
get_n_11 = partial(get_ndcg, top=11)
get_n_10 = partial(get_ndcg, top=10)


def get_inv_propensity(train_y: csr_matrix, a=0.55, b=1.5):
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)


def get_psp(
    prediction: TPredict,
    targets: TTarget,
    inv_w: np.ndarray,
    mlb: TMlb = None,
    classes: TClass = None,
    top=5,
):
    mlb = get_mlb(classes, mlb)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top]).multiply(inv_w)
    num = prediction.multiply(targets).sum()
    t, den = csr_matrix(targets.multiply(inv_w)), 0
    for i in range(t.shape[0]):
        den += np.sum(np.sort(t.getrow(i).data)[-top:])
    return num / den


get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)
get_psp_7 = partial(get_psp, top=7)
get_psp_9 = partial(get_psp, top=9)
get_psp_11 = partial(get_psp, top=11)
get_psp_10 = partial(get_psp, top=10)


def get_psndcg(
    prediction: TPredict,
    targets: TTarget,
    inv_w: np.ndarray,
    mlb: TMlb = None,
    classes: TClass = None,
    top=5,
):
    mlb = get_mlb(classes, mlb)
    log = 1.0 / np.log2(np.arange(top) + 2)
    psdcg = 0.0
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    for i in range(top):
        p = mlb.transform(prediction[:, i : i + 1]).multiply(inv_w)
        psdcg += p.multiply(targets).sum() * log[i]
    t, den = csr_matrix(targets.multiply(inv_w)), 0.0
    for i in range(t.shape[0]):
        num = min(top, len(t.getrow(i).data))
        den += -np.sum(np.sort(-t.getrow(i).data)[:num] * log[:num])
    return psdcg / den


get_psndcg_1 = partial(get_psndcg, top=1)
get_psndcg_3 = partial(get_psndcg, top=3)
get_psndcg_5 = partial(get_psndcg, top=5)
get_psndcg_7 = partial(get_psndcg, top=7)
get_psndcg_9 = partial(get_psndcg, top=9)
get_psndcg_11 = partial(get_psndcg, top=11)
get_psndcg_10 = partial(get_psndcg, top=10)
