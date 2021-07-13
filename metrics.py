import torch
import numpy as np
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

HMs = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]

def myMetrics(current, before, weight):
    current = np.asarray(current)
    before = np.asarray(before)
    weight = np.asarray(weight)
    weighted_diff_sum = np.sum((current - before)*weight)
    return weighted_diff_sum > 0

def roc_auc(preds, labels):
    scores = ["ROC"]
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    for ith in range(len(HMs)):
        scores.append(roc_auc_score(labels[:,ith], preds[:,ith]))
    return scores

def prc_auc(preds, labels):
    scores = ["PRC"]
    # recalls, precisions = ["Recall"], ["Precision"]
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    for ith in range(len(HMs)):
        precision, recall, thresholds = precision_recall_curve(labels[:,ith], preds[:,ith])
        scores.append(auc(recall, precision))
        # recalls.append(np.median(recall))
        # precisions.append(np.median(precision))
    return scores#, (recalls, precisions)