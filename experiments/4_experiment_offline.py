import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath("."))
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from src.RE_dFuzzStream import REdFuzzStreamSummarizer
from src.functions.merge import AllMergers
from src.functions.distance import EuclideanDistance
from src.functions.membership import FuzzyCMeansMembership
from src.functions import metrics


def get_dataset_params(datasetName):

    # datasetName :'Insects', 'RBF1_40000',  'Benchmark1_11000', 'Gaussian_4C2D800', 'NOAA', 'PowerSupply'

    dataset_params = {}

    if (datasetName == 'Benchmark1_11000'):
        # "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv"
        dataset_params["datasetPath"] = Path("datasets","Benchmark1_11000.csv")
        dataset_params["threshList"] = [0.8, 0.9, 0.25, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                      0.25, 0.5, 0.8, 0.25, 0.25, 0.65, 0.65, 0.8, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        dataset_params["numChunks"] = 11
        dataset_params["chunksize"] = 1000
        dataset_params["n_clusters"] = 2
        dataset_params["max_fmics"] = 50
        dataset_params["break_n"] = 11_000
    elif (datasetName == 'RBF1_40000'):
        # "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv"
        dataset_params["datasetPath"] = Path("datasets","RBF1_40000.csv")
        dataset_params["threshList"] = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                      0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        dataset_params["numChunks"] = 40
        dataset_params["chunksize"] = 1000
        dataset_params["n_clusters"] = 3
        dataset_params["max_fmics"] = 100
        dataset_params["break_n"] = 40_000
    elif (datasetName == 'Gaussian_4C2D800'):
        # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
        dataset_params["datasetPath"] = Path("datasets","DS1.csv")
        dataset_params["threshList"] = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                      0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        dataset_params["numChunks"] = 8
        dataset_params["chunksize"] = 100
        dataset_params["n_clusters"] = 4
        dataset_params["max_fmics"] = 100
        dataset_params["break_n"] = 800
    elif (datasetName == 'Insects'):
        # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
        dataset_params["datasetPath"] = Path("datasets","INSECTS-incremental_balanced_norm.csv")
        dataset_params["threshList"] = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                      0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        dataset_params["numChunks"] = 19
        dataset_params["chunksize"] = 3000
        dataset_params["n_clusters"] = 6
        dataset_params["max_fmics"] = 100
        dataset_params["break_n"] = 57_000
    elif (datasetName == 'PowerSupply'):
        # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
        dataset_params["datasetPath"] =  Path("datasets","powersupply.csv")
        dataset_params["threshList"] = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                      0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        dataset_params["numChunks"] = 29
        dataset_params["chunksize"] = 1000
        dataset_params["n_clusters"] = 24
        dataset_params["max_fmics"] = 100
        dataset_params["break_n"] = 29_000
    elif (datasetName == 'NOAA'):
        # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
        dataset_params["datasetPath"] =  Path("datasets","NEweather_norm.csv")
        dataset_params["threshList"] = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                      0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        dataset_params["numChunks"] = 18
        dataset_params["chunksize"] = 1000
        dataset_params["n_clusters"] = 2
        dataset_params["max_fmics"] = 100
        dataset_params["break_n"] = 18_000
    elif (datasetName == 'sensor'):
        # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
        dataset_params["datasetPath"] =  Path("datasets","sensor.csv")
        dataset_params["threshList"] = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                      0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        dataset_params["numChunks"] = 2219
        dataset_params["chunksize"] = 1000
        dataset_params["n_clusters"] = 54
        dataset_params["max_fmics"] = 100
        dataset_params["break_n"] = 100_000 # 2_219_000

    return dataset_params

def run_experiment(dataset_params, start=0, end=-1):
    sm = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    min_fmics = 5

    if end==-1:
        end = len(sm)

    outputPath = dataset_params["outputPath"]
    datasetPath = dataset_params["datasetPath"]
    chunksize = dataset_params["chunksize"]
    max_fmics = dataset_params["max_fmics"]
    threshList = dataset_params["threshList"]
    n_clusters = dataset_params["n_clusters"]
    break_n = dataset_params['break_n']

    if not outputPath.exists():
        outputPath.mkdir(parents=True,exist_ok=True)

    for vecIndex, simIDX in enumerate(sm[start:end]):
        threshIDX = threshList[start+vecIndex]
        summarizer = REdFuzzStreamSummarizer(
            max_fmics=max_fmics,
            distance_function=EuclideanDistance.distance,
            merge_threshold=threshIDX,
            merge_function=AllMergers[simIDX](simIDX, threshIDX, max_fmics),
            membership_function=FuzzyCMeansMembership.memberships,
            chunksize=chunksize,
            n_macro_clusters=n_clusters,
            time_gap=chunksize
        )

        timestamp = 0

        # Read files in chunks
        with pd.read_csv(datasetPath,
                         dtype={ "class": str},
                         chunksize=chunksize) as reader:
            ARI = []
            SIL = []
            for chunk in reader:
                log_text = (f"Summarizing examples from {timestamp} to "
                            f"{timestamp + chunksize-1} -> sim {simIDX} "
                            f"and thrsh {threshIDX}")

                for index, example in chunk.iterrows():
                    if timestamp > break_n:
                        break
                    # Summarizing example
                    ex_data = example[0:-1]
                    ex_class = example[-1]
                    summarizer.summarize(ex_data, ex_class, timestamp)
                    timestamp += 1

                    # Offline - Evaluation
                    if (timestamp) % summarizer.time_gap == 0:
                        ari, sil = metrics.offline_stats(summarizer, chunk)
                        purity = metrics.offline_purity(summarizer._Vmm)
                        ARI.append(ari)
                        SIL.append(sil)
                        print(simIDX,timestamp,ari,sil, purity)
                        with open(outputPath / f"dFuzz-{simIDX}-{threshIDX}.csv", mode='a') as res_file:
                            res_file.write(f"{outputPath.name},{simIDX},{threshIDX},{timestamp}, {ARI[-1]},{SIL[-1]}")

                if timestamp > break_n:
                    break
            print(f"{outputPath.name}: {simIDX},{threshIDX}: {np.mean(ARI)} {np.mean(SIL)}")


    print("--- End of execution --- ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    parser.add_argument('--start', type=int, default=0,
                        help='start (d = 0) - first measure in list')

    parser.add_argument('--end', type=int, default=-1,
                        help='end (d = -1) - last measure in list')

    parser.add_argument('--dataset', type=str, default='Gaussian_4C2D800',
                        help='Dataset: Benchmark1_11000 (d) or RBF1_40000')


    args = parser.parse_args()

    currentPath = Path.cwd()

    dataset_name = args.dataset
    start = args.start
    end = args.end
    dataset_params = get_dataset_params(dataset_name)
    dataset_params["datasetPath"] =  currentPath / dataset_params["datasetPath"]
    dataset_params["outputPath"] =  currentPath / "output" / dataset_name
    dataset_params["dtypes"] = {"class": str}
    # dataset_params["break_n"] = 100_000_000  # some dataset are not exact mutiples of chunk*R and it causes to fail, so I stopt it for the last few examples

    print(f"Start experiment for dataset {dataset_name} and start: {start} - end: {end}")
    run_experiment(dataset_params, start=start, end=end)
