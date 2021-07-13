import os
import pyBigWig
import numpy as np
from tqdm import tqdm
from Bio import SeqIO

HMs = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]
chr_map = {"chr1":"NC_000001.11", "chr2":"NC_000002.12", "chr3":"NC_000003.12", "chr4":"NC_000004.12", "chr5":"NC_000005.10", "chr6":"NC_000006.12", "chr7":"NC_000007.14", "chr8":"NC_000008.11", "chr9":"NC_000009.12", "chr10":"NC_000010.11", "chr11":"NC_000011.10", "chr12":"NC_000012.12", "chr13":"NC_000013.11", "chr14":"NC_000014.9", "chr15":"NC_000015.10", "chr16":"NC_000016.10", "chr17":"NC_000017.11", "chr18":"NC_000018.10", "chr19":"NC_000019.10", "chr20":"NC_000020.11", "chr21":"NC_000021.9", "chr22":"NC_000022.11", "chrX":"NC_000023.11"}

class CreateDataset():
    def __init__(self, biosample, dataset_dir, window_size=200, seq_length=2000, HMs=HMs, chr_map=chr_map):
        self.biosample = biosample
        print("Currently processing :", biosample)

        self.window_size = window_size
        self.seq_length = seq_length
        self.HMs = HMs
        self.chr_map = chr_map
        
        self.whole_dna = SeqIO.to_dict(SeqIO.parse(dataset_dir + "GRCh38_latest_genomic.fna", "fasta"))
        self.meth = self.readFile("{}/{}/methylation/".format(dataset_dir, biosample), "bigBed")
        self.peaks = dict({(HM, None) for HM in self.HMs})

        for HM in self.HMs:
            self.peaks[HM] = self.readFile("{}/{}/{}/".format(dataset_dir, biosample, HM), "bigBed")
        print("Reading file finished ......")
        
    def readFile(self, dataset_dir, data_type):
        file_name = [f for f in os.listdir(dataset_dir) if f[-len(data_type):] == data_type][0]
        return pyBigWig.open(dataset_dir + file_name)

    def extractDnaSignal(self, seq, chr_ith, start_pos):
        output = []
        row_idx = {"A":0, "T":1, "C":2, "G":3}
        chr_ith = self.chr_map[chr_ith]
        for base in seq:
            temp = [0]*4
            temp[row_idx[base]] = 1
            output.append(temp)
        return np.asarray(output, dtype=np.float32).T

    def extractMethSignal(self, chr_ith, start_pos, single_strand=None):
        output = [0] * self.seq_length
        end_pos = start_pos + self.seq_length
        meths = self.meth.entries(chr_ith, start_pos, end_pos)
        if meths == None:
            return np.array(output)
        for meth in self.meth.entries(chr_ith, start_pos, end_pos):
            signal = meth[2].split("\t")
            strand, meth_pos, meth_per = signal[2], int(signal[3]), int(signal[-1])/100
            if single_strand:
                if (strand == single_strand) and (meth_per != 0):
                    output[meth_pos-start_pos] = meth_per
            else:
                if meth != 0:
                    output[meth_pos-start_pos] = meth_per
        return np.asarray(output, dtype=np.float32)

    def getLabelVector(self, chr_ith, win_start_pos):
        label = [0] * len(self.HMs)
        win_end_pos = win_start_pos + self.window_size
        signals = [(idx, self.peaks[HM].entries(chr_ith, win_start_pos, win_end_pos)) for idx, HM in enumerate(self.HMs)]
        signals = filter(lambda x: x[1] != None, signals)
        for idx, signal in signals:
            if len(signal) == 1:
                peak_start_pos, peak_end_pos = signal[0][0], signal[0][1]
                if win_start_pos >= peak_start_pos:
                    if win_end_pos <= peak_end_pos:
                        label[idx] = 1 # Front case
                    elif (peak_end_pos-win_start_pos) >= self.window_size//2:
                        label[idx] = 1 # Between case
                else:
                    if win_end_pos <= peak_end_pos:
                        if (win_end_pos-peak_start_pos) >= self.window_size//2:
                            label[idx] = 1 # Behind case
                    elif (peak_end_pos-peak_start_pos) >= self.window_size//2:
                        label[idx] = 1 # Inter case
            else: # Two peaks case
                peak_f_end_pos = signal[0][1]
                peak_b_start_pos = signal[1][0]
                if (peak_f_end_pos-peak_b_start_pos) <= self.window_size//2:
                    label[idx] = 1
                if len(signal) > 2:
                    print("warning")
        return np.asarray(label, dtype=np.float32)

    def getAllPeakStartPosPair(self, chr_ith):
        start_pos_and_length = []
        leng = self.meth.chroms(chr_ith)
        for HM in self.HMs:
            peak = self.peaks[HM].entries(chr_ith, 0, leng)
            for p in peak:
                start_pos, length = int(p[0]), int(p[1])-int(p[0])
                if length >= self.window_size//2:
                    start_pos_and_length.append((start_pos, length))
        return sorted(start_pos_and_length, key=lambda x: x[0])

    def getEntry(self, chr_ith, start_pos):
        seq_start_pos = start_pos - (self.seq_length-self.window_size)//2
        seq_end_pos = seq_start_pos + self.seq_length
        dna_seq = self.whole_dna[self.chr_map[chr_ith]].seq[seq_start_pos:seq_end_pos].upper()
        check_point = set(dna_seq)
        if len(check_point) != 4:
            return # Check whether N in dna sequence
        dna = self.extractDnaSignal(dna_seq, chr_ith, seq_start_pos)
        meth = self.extractMethSignal(chr_ith, seq_start_pos)
        label = self.getLabelVector(chr_ith, start_pos)
        scope = [seq_start_pos, seq_end_pos]
        return [scope, dna, meth, label]

    def main(self, output_dir):
        for ith in list(range(1,23))+["X"]:
            current_pos = 0
            chr_ith = "chr" + str(ith)
            dataset = [[] for _ in range(4)]
            peak_length_pair = self.getAllPeakStartPosPair(chr_ith)

            for start_pos, length in tqdm(peak_length_pair):
                if start_pos >= current_pos:
                    times = 1
                    if length >= 200: # Sanning all of peak signals with window
                        start_pos = start_pos - self.window_size//2
                        times += (length-self.window_size//2) // self.window_size
                        times += ((length-self.window_size//2) % self.window_size) // (self.window_size//2)
                    for _ in range(times):
                        entry = self.getEntry(chr_ith, start_pos)
                        if entry:
                            [dataset[idx].append(entry[idx]) for idx in range(4)]
                        start_pos += self.window_size
                    current_pos = start_pos
                else:
                    continue

            training_dir = "{}/{}/".format(output_dir, self.biosample)
            if not os.path.isdir(training_dir):
                os.mkdir(training_dir)
            print(" Starting saving {}--{} ".format(self.biosample, chr_ith).center(80, "="))
            training_path = "{}/{}.npz".format(training_dir, chr_ith)
            np.savez_compressed(training_path, scope=dataset[0], dna=dataset[1], meth=dataset[2], label=dataset[3])

if __name__ == "__main__":
    # biosamples = ["A549", "GM12878", "K562"]
    dataset_dir = "../dataset/new/"
    output_dir = "../dataset/training/"
    biosample = "H1"

    creator = CreateDataset(biosample, dataset_dir)
    creator.main(output_dir)