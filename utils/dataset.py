import os
import time
import pyBigWig
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager

class Dataset():
    def __init__(self, name, dataset_dir, bin_size):
        self.name = name
        self.bin_size = bin_size
        self.path = dataset_dir
        self.non_bin = False

        initial = dict([("chr"+str(ith),[]) for ith in list(range(1,23))+["X"]])
        self.HMs = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]
        # self.HMs = ["H3K9me3"]
        self.signals = dict([(HM, initial.copy())for HM in self.HMs])

        if bin_size == 1:
            self.non_bin = True
            self.bin_size = 10000

        ######## Automatic do processing code

    def getDatasetPath(self, HM):
        dataset_path = self.path + HM + "/"
        dataset_path += os.listdir(dataset_path)[0]
        return dataset_path

    def ruleOfFill(self, idx, signal):
        # Check if first position is nan
        if (idx == 0) and np.isnan(signal[idx+1]):
            return 0
        elif (idx == 0) and (not np.isnan(signal[idx+1])):
            return signal[idx+1]
        # Check if final position or next postion is nan
        elif (idx == (len(signal)-1)) or (np.isnan(signal[idx+1])):
            return signal[idx-1]
        # If previous and next position are not nan, sum these values and average
        else:
            return (signal[idx-1]+signal[idx+1]) / 2

    def processMissingValue(self, signal):
        if not all(signal == signal):
            for idx in range(len(signal)):
                if np.isnan(signal[idx]):
                    processed = self.ruleOfFill(idx, signal)
                    signal[idx] = processed

        return signal if self.non_bin else np.mean(signal)

    def getBinSignalValuePerChrom(self, HM, chr_ith):
        # Becsue bigwigfile object cannot be pickled, we just open dataset in each processes
        bw = pyBigWig.open(self.getDatasetPath(HM))
        chr_signal = np.array([], dtype=np.float32)
        chr_len = bw.chroms()[chr_ith]

        for end_idx in range(self.bin_size, chr_len, self.bin_size):
            start_idx = end_idx - self.bin_size
            raw_signal = bw.values(chr_ith, start_idx, end_idx, numpy=True)
            processed_signal = self.processMissingValue(raw_signal)

            if self.non_bin:
                chr_signal = np.append(chr_signal, processed_signal)
            else:
                chr_signal = np.append(chr_signal, [processed_signal])
        else:
            raw_signal = bw.values(chr_ith, end_idx, chr_len, numpy=True)
            processed_signal = self.processMissingValue(raw_signal)
            
            if self.non_bin:
                chr_signal = np.append(chr_signal, processed_signal)
            else:
                chr_signal = np.append(chr_signal, [processed_signal])
            # print("--- {} finished".format(chr_ith))

        return (chr_ith, chr_signal)

    def mp_preprocess(self):
        progress_bar = tqdm(self.HMs)
        results = dict()

        for HM in progress_bar:
            # print("=== {} ==============".format(HM))
            progress_bar.set_description("Processing {}--{} ".format(self.name, HM))

            with Pool() as pool:
                args = [(HM, "chr"+str(ith)) for ith in list(range(1,23))+["X"]]
                result = pool.starmap_async(self.getBinSignalValuePerChrom, args)

                # Prevents any more tasks from being submitted to the pool.
                # Once all the tasks have been completed the worker processes will exit.
                pool.close()
                # Wait for the worker processes to exit.
                # One must call close() or terminate() before using join() .
                pool.join()

            # To solve the problem that multi-process run more and more slowly
            results[HM] = result.get()

        # Fill signal to my data format
        for HM in self.HMs:
            for chr_ith, signal in results[HM]:
                self.signals[HM][chr_ith] = signal

         
if __name__ == "__main__":
    start = time.time()

    A549 = Dataset("A549", "dataset/cell line/A549/", 1)
    A549.mp_preprocess()

    end = time.time()

    print("Total signal length :", 3031042417)
    print(len(A549.signals["H3K4me3"]["chr1"]))
    print("Time :", end-start)
