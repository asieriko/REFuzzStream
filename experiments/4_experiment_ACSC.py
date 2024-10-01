#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:56:54 2023

@author: asier
"""
from pathlib import Path
import os
import sys #/data/asier/Ikerketa/Projects/DataStream/experiments'
# sys.path.append(os.path.abspath( "..")) # '/data/asier/Ikerketa/Projects'
# sys.path.append(os.path.abspath( "../..")) # '/data/asier/Ikerketa'
# #sys.path.append("/home/au24677/DataStream")
# sys.path.append(Path.cwd().parent) # PosixPath('/data/asier/Ikerketa/Projects')
# sys.path.append(Path.cwd()) # PosixPath('/data/asier/Ikerketa/Projects/DataStream')
sys.path.append(os.path.abspath( "."))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scluster.ACSC.ACSC import ACSC
from get_datasets import get_dataset_params


def main(dataset_params, nSamples_set=[2,3,5,10], sleepMax_set=[1, 2, 3, 4, 6], epsilon_set=[0.01, 0.1, 0.5, 0.75, 1]):

    outputPath = dataset_params["outputPath"]
    datasetPath = dataset_params["datasetPath"]
    dtypes = dataset_params["dtypes"]
    chunksize = dataset_params["chunksize"]
    R = dataset_params["numChunks"]
    break_n = dataset_params["break_n"]

    alpha = dataset_params["alpha"]
    for nSamples in nSamples_set:
        for sleepMax in sleepMax_set:
            for epsilson in epsilon_set:
                run_ACSC(datasetPath, outputPath, dtypes, chunksize, nSamples, sleepMax, epsilon, break_n)

def run_ACSC(dataset, outputPath, dtypes, chunksize, nSamples, sleepMax, epsilon, break_n):
    acsc = ACSC(nSamples, sleepMax, epsilon) 
    name = f"ACSC {dataset.name}-{nSamples}-{sleepMax}-{epsilon}"
    print(f"ACSC for {dataset=}")
    
    # Read files in chunks
    with pd.read_csv(dataset,
                     # names=['X1','X2','class'],  # For Gaussian dataset only
                     dtype=dtypes,
                     chunksize=chunksize) as reader:
        timestamp = 0
        ARI = []
        SIL = []
        for chunk in reader:
            print(f"Summarizing examples from {timestamp} to {timestamp + chunksize - 1}")
            X = chunk.iloc[:,:-1].to_numpy()
            y = chunk.iloc[:,-1].fillna("-1").to_numpy()
            acsc.process_window(X)  # FIXME: remove class          
            
            final_clusters = acsc.clusters
            print(f"{len(final_clusters)} Clusters found.")

            labels_ASCS = np.array([acsc.predict(x) for x in X])

            label_real = y
            label_real[label_real == np.nan] = -1
            ARI.append(adjusted_rand_score(label_real[-chunksize:], labels_ASCS[-chunksize:]))
            if len(X) - 1 >= len(np.unique(labels_ASCS)) > 1:
                SIL.append(silhouette_score(X, labels_ASCS))
            else:
                SIL.append(np.nan)
            print(ARI[-1], SIL[-1])
            # plot(X, labels_tsf, M, labels_tsf==-1,ranges=[[0,0],[1,1]],
            #      title=f"{name}-\n-{timestamp}",file_name=f"{outputPath / ({name}-{timestamp}).png")
            with open(f"{outputPath}-{nSamples}-{sleepMax}-{epsilon}.csv",mode='a') as res_file:
                res_file.write(f"{dataset.name},{nSamples},{sleepMax},{epsilon},{timestamp}, {ARI[-1]},{SIL[-1]}")
            
            timestamp += chunksize
            
        print(np.mean(ARI))
        print(np.mean(SIL))
        with open(f"{outputPath}-{dataset.name}-{nSamples}-{sleepMax}-{epsilon}-final.txt",mode='a') as res_file:
                res_file.write(f"{np.mean(ARI)}") #,{SIL[-1]}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    parser.add_argument('--nSamples', type=int, default=2,
                        help='nSamples (d = 2)') 

    parser.add_argument('--sleepMax', type=int, default=2,
                        help='sleepMax (d = 2)') 

    parser.add_argument('--epsilon', type=float, default=0.3,
                        help='epsilon (d = 0.5)') 

    parser.add_argument('--dataset', type=str, default='NOAA',
                        help='Dataset: Benchmark1_11000 (d) or RBF1_40000')

    args = parser.parse_args()

    currentPath = Path.cwd()

    dataset_name = args.dataset
    # dataset: = "INSECTS-incremental_balanced_norm.csv", "Benchmark1_11000.csv" , "RBF1_40000.csv"
    # Gaussian_4C2D800  PowerSupply NOAA RBF1_40000 Benchmark1_11000
    dataset_params = get_dataset_params(dataset_name)

    dataset_params["datasetPath"] =  currentPath / dataset_params["datasetPath"]
    dataset_params["outputPath"] =  currentPath / "output" / "ACSC" / dataset_name.split(".")[0]
    dataset_params["dtypes"] = {"class": str}
    # dtypes = {"X1": float, "X2": float, "class": str}



    nSamples = [args.nSamples]
    sleepMax = [args.sleepMax]
    epsilon = [args.epsilon]
    main(dataset_params,nSamples,sleepMax,epsilon)
