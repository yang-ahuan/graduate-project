import torch
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import Dataset, IterableDataset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Ignore sklearn duplicate warning
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

class TestDataset(Dataset):
    def __init__(self, dataset_path, run_type):
        with np.load(dataset_path) as data:
            dna = np.asarray(data["dna"], dtype=np.float32)
            meth = np.asarray(data["meth"], dtype=np.float32)
            label = np.asarray(data["label"], dtype=np.float32)

        self.run_type = run_type
        self.index = self.getIndex(len(dna))
        self.dna = torch.from_numpy(dna[self.index])
        self.meth = torch.from_numpy(meth[self.index])
        self.label = torch.from_numpy(label[self.index])
        if self.run_type == "train":
            self.pos_weight = self.getPosWeight()

    def getIndex(self, amount):
        index = np.arange(amount)
        if self.run_type == "train":
            return index[:int(amount*0.9)]
        elif self.run_type == "valid":
            return index[int(amount*0.9):]

    def getPosWeight(self):
        pos_count = torch.sum(self.label, dim=0)
        neg_count = len(self.label) - pos_count
        return neg_count / pos_count

    def __getitem__(self, index):
        dna_feat = self.dna[index]
        meth_feat = torch.reshape(self.meth[index], (1,-1))
        return torch.cat((dna_feat, meth_feat)), self.label[index]

    def __len__(self):
        return len(self.label)

class HalfDataset(Dataset):
    def __init__(self, dataset_dir, chr_iths, run_type):
        self.chr_iths = chr_iths
        self.dna = self.getData(dataset_dir, "dna")
        self.meth = self.getData(dataset_dir, "meth")
        self.label = self.getData(dataset_dir, "label")

        if run_type == "train":
            self.pos_weight = self.getPosWeight()

    def getData(self, dataset_dir, category):
        temp = []
        for ith in self.chr_iths:
            dataset_path = dataset_dir + "chr{}.npz".format(ith)
            with np.load(dataset_path) as data:
                temp.append(np.asarray(data[category], dtype=np.float32))
        return torch.from_numpy(np.concatenate(temp))

    def getPosWeight(self):
        pos_count = torch.sum(self.label, dim=0)
        neg_count = len(self.label) - pos_count
        return neg_count / pos_count

    def __getitem__(self, index):
        dna_feat = self.dna[index]
        meth_feat = torch.reshape(self.meth[index], (1,-1))
        return torch.cat((dna_feat, meth_feat)), self.label[index]

    def __len__(self):
        return len(self.label)

class WholeDataset(Dataset):
    def __init__(self, dataset_dir, biosample):
        self.dna, self.meth, self.label = None, None, None
        
        for ith in list(range(1,23))+["X"]:
            print("Loading chr{} ... ...".format(ith))
            dataset_path = dataset_dir + "{}/chr{}.npz".format(biosample, ith)
            with np.load(dataset_path) as data:
                self.dna = data["dna"] if self.dna is None else np.concatenate((self.dna, data["dna"]), dtype=np.float32)
                self.meth = data["meth"] if self.meth is None else np.concatenate((self.meth, data["meth"]), dtype=np.float32)
                self.label = data["label"] if self.label is None else np.concatenate((self.label, data["label"]), dtype=np.float32)

        self.dna = torch.from_numpy(self.dna)
        self.meth = torch.from_numpy(self.meth)
        self.label = torch.from_numpy(self.label)

    def __getitem__(self, index):
        dna_feat = self.dna[index]
        meth_feat = torch.reshape(self.meth[index], (1,-1))
        return torch.cat((dna_feat, meth_feat)), self.label[index]

    def __len__(self):
        return len(self.label)

class RemixDataset(IterableDataset): # Dataloader will transform data to tensor type
    def __init__(self, dataset_dir, scope, biosamples, batch_size=None, dna=True, meth=True):
        self.scope = scope
        self.meth = meth
        self.dna = dna
        self.batch_size = batch_size
        self.biosamples = biosamples
        self.dataset_dir = dataset_dir

    def loadData(self, ith_part):
        temp_input, temp_label = None, None
        for biosample in self.biosamples:
            dataset_path = self.dataset_dir+"{}/remix_stratified/partition_{}.npz".format(biosample, ith_part)
            with np.load(dataset_path) as data:
                if self.meth and self.dna:
                    input = np.concatenate((data["dna"],np.reshape(data["meth"],(-1,1,2000))), axis=1)
                elif self.dna:
                    input = data["dna"]
                elif self.meth:
                    input = np.reshape(data["meth"],(-1,1,2000))
                label = data["label"]
            if temp_input is None:
                temp_input = input
                temp_label = label
            else:
                temp_input = np.concatenate((temp_input, input))
                temp_label = np.concatenate((temp_label, label))
        return temp_input.astype(np.float32), temp_label.astype(np.float32)

    def main(self):
        for ith_part in self.scope:
            print("Start {}th partition ... ... ".format(ith_part+1))
            input, label = self.loadData(ith_part)
            if self.batch_size:
                n_splits = len(label) // self.batch_size
                mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True)#, random_state=0)
                for _, index in mskf.split(input, label):
                    for pair in zip(input[index], label[index]):
                        yield pair
            else:
                input, label = shuffle(input, label, random_state=64)
                for pair in zip(input, label):
                    yield pair

    def __iter__(self):
        return iter(self.main())