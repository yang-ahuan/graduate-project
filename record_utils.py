import os
import wandb
import torch
import pickle
from metrics import myMetrics

HMs = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]

def setSummary(min_valid, min_train, record):
    wandb.summary.update({"valid_loss": min_valid})
    wandb.summary.update({"train_loss": min_train})
    for idx, (roc, prc) in enumerate(record, start=-1):
        if idx==-1: continue
        wandb.summary.update({HMs[idx]+"_roc": roc})
        wandb.summary.update({HMs[idx]+"_prc": prc})

def save(parameter, log, dir, rank):
    torch.save(parameter, dir+"/{}.model".format(rank))
    with open(dir+"/{}_scores.pkl".format(rank), "wb") as f:
        pickle.dump(log, f)

def readPrcScore(dir, files):
    file = [f for f in files if f[-3:]=="pkl"][0]
    with open(dir+"/"+file, "rb") as f:
        scores = pickle.load(f)
    return scores

def recordBestModel(parameter, current_log, dir, weight):
    ranks = ["top_1", "top_2", "top_3"]
    for rank in ranks:
        path = dir + rank
        files = os.listdir(path)
        if files == []:
            save(parameter, current_log, path, rank)
            break
        else:
            before_log = readPrcScore(path, files)
            before_prc = [before_log[HM+"_prc"] for HM in HMs]
            current_prc = [current_log[HM+"_prc"] for HM in HMs]
            if myMetrics(current_prc, before_prc, weight):
                save(parameter, current_log, path, rank)
                break
            