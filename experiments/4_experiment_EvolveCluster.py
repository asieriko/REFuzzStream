#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:56:54 2023

@author: asier
"""
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath( "."))
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scluster.EvolveCluster import EvolveCluster
from scluster.EvolveCluster.functions import calculate_distances
from get_datasets import get_dataset_params


def main(dataset_params, tau_set=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):

    outputPath = dataset_params["outputPath"]
    datasetPath = dataset_params["datasetPath"]
    dtypes = dataset_params["dtypes"]
    chunksize = dataset_params["chunksize"]
    numChunks = dataset_params["numChunks"]
    break_n = dataset_params["break_n"]

    alpha = dataset_params["alpha"]
    for tau in tau_set:
        run_EvolveCluster(datasetPath, outputPath, dtypes, chunksize, numChunks, tau, break_n)


def run_EvolveCluster(dataset, outputPath, dtypes, chunksize, numChunks, tau, break_n):
    name = f"EvolveCluster {dataset.name}-{tau}"
    
    # Read files in chunks
    first = True
    with pd.read_csv(dataset,
                     # names=['X1','X2','class'],  # For Gaussian dataset only
                     dtype=dtypes,
                     chunksize=chunksize) as reader:
        timestamp = 0
        ARI = []
        SIL = []
        for chunk in reader:
            print(f"Summarizing examples from {timestamp} to {timestamp + chunksize - 1}")
            
            if first:
                #def generic_csv(input_file, dimension, chunk_size, tau):
                data = chunk.copy() # FIXME
                data['clusters'] = np.nan
                tmp = data.iloc[:,:-2].to_numpy(copy=True)

                D = calculate_distances(tmp)
                medoids = []
                for i in range(1,3):
                    test = data[data['class'] == str(i)]
                    D_test = calculate_distances(test.iloc[:,:-2].to_numpy(copy=True))
                    medoid = np.argmin(D_test.sum(axis=0))
                    medoid = int(test.iloc[medoid].name)
                    data.loc[data['class'] == str(i), 'clusters'] = medoid
                    medoids.append(medoid)

                a = data['clusters'].to_list()
                C = {}
                for i in range(len(medoids)):
                    C[str(i)] = []
                    for j in range(len(a)):
                        if a[j] == medoids[i]:
                            C[str(i)].append(j)
                    C[str(i)] = np.array(C[str(i)])

                ec = EvolveCluster.EvolveCluster(data.iloc[:,:-2].to_numpy(copy=True), C, medoids, D, tau)
                first = False
            else:
                ec.cluster([chunk.iloc[:,:-1].to_numpy(copy=True)])
                
            print(f"{len(ec.clusters)} Clusters found.")  #ec.cetnroids

            # Update to fit ec.
            labels_ec = []
            for i in range(len(chunk)):
                assigned = False
                for j in ec.clusters[0].keys():
                    if i in ec.clusters[0][j]:
                        labels_ec.append(int(j))
                        assigned = True
                        break
                if not assigned:
                    labels_ec.append(-1)

            label_real = chunk.iloc[:,-1].copy()
            label_real[label_real.isnull()] = -1
            ARI.append(adjusted_rand_score(label_real.iloc[-chunksize:].astype('int'), labels_ec[-chunksize:]))
            # if len(label_real) - 1 >= len(np.unique(labels_ec)) > 1:  # FIXME: Check
            #     sc = silhouette_score(label_real, labels_ec)
            #     SIL.append(sc)
            # else:
            #     SIL.append(np.nan)
            # print(ARI[-1], SIL[-1])
            print(ARI[-1])
            # plot(X, labels_tsf, M, labels_tsf==-1,ranges=[[0,0],[1,1]],
            #      title=f"{name}-\n-{timestamp}",file_name=f"{outputPath / ({name}-{timestamp}).png")
            with open(f"{outputPath}-{tau}.csv",mode='a') as res_file:
                res_file.write(f"{dataset.name},{tau},{timestamp}, {ARI[-1]}")#,{SIL[-1]}")
            
            timestamp += len(chunk)

        print(np.mean(ARI))
        # print(np.mean(SIL))
        with open(f"{outputPath}-{dataset.name}-{tau}-final.txt",mode='a') as res_file:
                res_file.write(f"{np.mean(ARI)}") #,{SIL[-1]}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    parser.add_argument('--tau', type=float, default=0.01,
                        help='tau (d = 0.01)') 

    parser.add_argument('--dataset', type=str, default='RBF1_40000',
                        help='Dataset: Benchmark1_11000 (d) or RBF1_40000')

    args = parser.parse_args()

    currentPath = Path.cwd()

    dataset_name = args.dataset
    # dataset: = "INSECTS-incremental_balanced_norm.csv", "Benchmark1_11000.csv" , "RBF1_40000.csv"
    # Gaussian_4C2D800  PowerSupply
    dataset_params = get_dataset_params(dataset_name)

    dataset_params["datasetPath"] =  currentPath / dataset_params["datasetPath"]
    dataset_params["outputPath"] =  currentPath / "output" / "EvolveCluster" / dataset_name.split(".")[0]
    dataset_params["dtypes"] = {"class": str}
    # dtypes = {"X1": float, "X2": float, "class": str}

    tau= [args.tau]
    main(dataset_params,tau)
