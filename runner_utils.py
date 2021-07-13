import wandb
import torch
import pickle
import numpy as np
from metrics import *
from record_utils import *
from tabulate import tabulate

HMs = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]

def getPosCountAndTotal(biosamples):
    total, pos_count = 0, np.zeros(6)
    with open("dataset/training/dist_whole_dataset.pkl", "rb") as f:
        dist = pickle.load(f)
    for biosample in biosamples:
        total += dist[biosample]["total"]
        pos_count += np.array([dist[biosample][HM] for HM in HMs])
    return pos_count, total

def getClassWeight(config):
    pos_count, total = getPosCountAndTotal(config.biosample)
    class_weight = total / (2*pos_count)
    if "update" in dir(config):
        if config.class_weight == True: config.update({"class_weight":class_weight}, allow_val_change=True)
        if config.pos_weight == True: config.update({"pos_weight":class_weight}, allow_val_change=True)
    return torch.as_tensor(class_weight)

def getPosWeight(config):
    pos_count, total = getPosCountAndTotal(config.biosample)
    pos_weight = (total-pos_count) / pos_count
    if config.pos_weight == True: config.update({"pos_weight":pos_weight}, allow_val_change=True)
    return torch.as_tensor(pos_weight)

def getCriterion(config):
    # pos_weight = getPosWeight(config).to(config.device) if config.pos_weight else config.pos_weight
    pos_weight = getClassWeight(config).to(config.device) if config.pos_weight else config.pos_weight
    class_weight = getClassWeight(config).to(config.device) if config.class_weight else config.class_weight
    return getattr(torch.nn, config.criterion)(weight=class_weight, pos_weight=pos_weight)

def getOptimizer(parameters, config):
    if config.optimizer == "Adam":
        return getattr(torch.optim, config.optimizer)(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "SGD":
        return getattr(torch.optim, config.optimizer)(parameters, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise Exception("Invalid optimizer name")

def eval(valid_loader, model, config):
    model.eval()
    valid_loss, preds, labels = [], [], []
    criterion = getCriterion(config)
    for x, y in valid_loader:
        x = x.to(config.device)
        y = y.to(config.device)
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
        valid_loss.append(loss.item())
        preds.append(pred.cpu())
        labels.append(y.cpu())
    roc_auc_score = roc_auc(preds, labels)
    prc_auc_score = prc_auc(preds, labels)
    return np.mean(valid_loss), roc_auc_score, prc_auc_score #, detail

def train(train_loader, valid_loader, model, config, saved_dir):
    criterion = getCriterion(config)
    optimizer = getOptimizer(model.parameters(), config)
    wandb.watch(model, log=None)

    best_prc_score = np.zeros(6)
    for epoch in range(config.epochs):
        train_loss = []
        print(" Epoch {} ".format(epoch+1).center(60,"="))
        for ith, (x, y) in enumerate(train_loader, start=1):
            x = x.to(config.device)
            y = y.to(config.device)

            model.train()
            pred = model(x)
            loss = criterion(pred, y)
            batch_loss = loss.detach().cpu().item()
            train_loss.append(batch_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)
        valid_loss, roc_score, prc_score = eval(valid_loader, model, config) # (recall, precision)
        log_file = {"valid_loss": valid_loss, "train_loss":train_loss,
                    "H3K4me3_roc": roc_score[1], "H3K4me3_prc": prc_score[1],
                    "H3K27ac_roc": roc_score[2], "H3K27ac_prc": prc_score[2],
                    "H3K4me1_roc": roc_score[3], "H3K4me1_prc": prc_score[3],
                    "H3K36me3_roc": roc_score[4], "H3K36me3_prc": prc_score[4],
                    "H3K9me3_roc": roc_score[5], "H3K9me3_prc": prc_score[5],
                    "H3K27me3_roc": roc_score[6], "H3K27me3_prc": prc_score[6]}
    
        # Record best status
        curr_prc_score = prc_score[1:]
        weight = config.pos_weight if config.pos_weight else getClassWeight(config)
        if myMetrics(curr_prc_score, best_prc_score, weight) > 0:
            record = [roc_score, prc_score]
            best_prc_score = curr_prc_score
            best_valid, best_train = valid_loss, train_loss
            if saved_dir: recordBestModel(model.state_dict(), log_file, saved_dir, weight)
        # Upload epoch status
        wandb.log(log_file)
    
        print("-- All batch : train loss -- {:.6f}, valid loss -- {:.6f}".format(train_loss, valid_loss))
        print(tabulate([roc_score, prc_score], HMs, floatfmt=".3f"))

    print("="*80)
    print("Best valid loss -- {:.6f}".format(best_valid))
    print(tabulate([record[0], record[1]], HMs, floatfmt=".3f"))
    print(" Finished ".center(60, "="))
    setSummary(best_valid, best_train, zip(*record))
            


            

