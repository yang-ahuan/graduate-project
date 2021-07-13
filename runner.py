from model import *
from dataset import *
from runner_utils import train

import wandb
import torch
import random
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    config = {
    "device" : device,
    "biosample": ["K562"], # List type
    # "train_chr": [1,3,5,7,9,11,13,15,17,19,21,"X"],
    # "valid_chr": [2,4,6],
    "epochs" : 20,
    "batch_size": 300,
# ===== Loss function para =====
    "criterion" : "BCEWithLogitsLoss",
    "pos_weight": True, # True, None
    "class_weight": None, # True, None
# ===== Optimizer para =========
    "optimizer" : "SGD",
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 0
    }

    wandb.init(project='data-impute-for-histone-modification', config=config)
    config = wandb.config

    print("Current computational resource :", config.device)
    # ========================= Test ===============================
    # print("Loading data ... ...")
    # dataset_path = "dataset/training/toyset.npz"
    # train_set = TestDataset(dataset_path, "train")
    # valid_set = TestDataset(dataset_path, "valid")
    # ========================= Half ===============================
    # dataset_dir = "dataset/training/{}/".format(*config.biosample)
    # print("Loading data ... ...")
    # train_set = HalfDataset(dataset_dir, config.train_chr, "train")
    # print("Loading data ... ...")
    # valid_set = HalfDataset(dataset_dir, config.valid_chr, "valid")
    # ========================= Remix ==============================
    dataset_dir = "dataset/training/"
    scope_order = list(range(10))
    random.Random().shuffle(scope_order)
    # Adding batch size para. means that using stratified mini-batch
    train_set = RemixDataset(dataset_dir, scope_order[:9], config.biosample, config.batch_size, dna=False)
    valid_set = RemixDataset(dataset_dir, scope_order[9:], config.biosample, dna=False)#, config.batch_size)
    # ==============================================================
    train_loader = DataLoader(train_set, batch_size=config.batch_size)#, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size)

    ################# Remember to change directory ##################
    saved_dir = "saved/sea_meth_K562/" # If we don't want to save model, tune this variable to None
    model = DeepSEA().to(config.device)
    train(train_loader, valid_loader, model, config, saved_dir)