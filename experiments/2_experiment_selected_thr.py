#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:08:55 2024

@author: asier.urio
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import sys
sys.path.append(os.path.abspath("."))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.RE_dFuzzStream import REdFuzzStreamSummarizer
from src.functions.merge import FuzzyDissimilarityMerger
from src.functions.merge import AllMergers
from src.functions.distance import EuclideanDistance
from src.functions.membership import FuzzyCMeansMembership
from src.functions import metrics


def experiment(dataset, chunksize=1000, min_fmics=5, max_fmics=100,start=0,end=0,fname="experiment",Figures=False):
    color = {'1': 'Red', '2': 'Blue', '3': 'Green', 'nan': 'gray', '-1':'gray'}  # For plotting
    sm = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    if end  == 0:
        end = len(sm) + 1
    
    if (dataset == 'Benchmark1_11000'):
        datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv"
        threshList = [0.9, 0.9, 0.25, 0.9, 0.5, 0.9, 0.8, 0.25, 0.25, 0.1, 0.25, 0.25, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.8, 0.5, 0.65, 0.65, 0.8, 0.9, 0.9, 0.8, 0.9, 0.25, 0.25, 0.25, 0.25, 0.25]
        numChunks = int(11_000 / chunksize)
        n_clusters = 2
    elif (dataset == 'RBF1_40000'):
        datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv"
        threshList = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        numChunks = int(40_000 / chunksize)
        n_clusters = 3
    elif (dataset == 'Insects'):
        datasetPath = "../datasets/INSECTS-incremental_balanced_norm.csv"  # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
        threshList = [0.8, 0.9, 0.25, 0.65, 0.8, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                      0.25, 0.65, 0.8, 0.5, 0.5, 0.65, 0.8, 0.9, 0.65, 0.65, 0.25, 0.25, 0.25, 0.25, 0.25]
        numChunks = int(57_000 / chunksize)
        n_clusters = 6

    # currentdir = os.path.dirname(os.path.realpath(__file__))
    # parentdir = os.path.dirname(currentdir)
    # sys.path.append(parentdir)
    output_path = Path.cwd() / "output"/ dataset
    Path(output_path).mkdir(parents=True,exist_ok=True)
    
    tabRes = pd.DataFrame(np.zeros((32, (numChunks * 3) + 3)))
    
    for vecIndex, simIDX in enumerate(sm[start:end]):
        threshIDX = threshList[vecIndex]
        df = pd.DataFrame(columns=['Chunk', 'Purity', 'pCoefficient',
                                   'pEntropy', 'XieBeni', 'MPC',
                                   'FukuyamaSugeno_1', 'FukuyamaSugeno_2'])
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
    
        summary = {'x': [], 'y': [], 'radius': [], 'color': [], 'weight': [], 'class': []}
        timestamp = 0
    
            
        # Read files in chunks
        with pd.read_csv(datasetPath,
                         dtype={"X1": float, "X2": float, "class": str},
                         chunksize=chunksize) as reader:
            for chunk in reader:
                log_text = (f"Summarizing examples from {timestamp} to "
                            f"{timestamp + chunksize-1} -> sim {simIDX} "
                            f"and thrsh {threshIDX}")
                print(log_text)
            
                for index, example in chunk.iterrows():
                    # Summarizing example
                    ex_data = example[0:-1]
                    ex_class = example[-1]
                    summarizer.summarize(ex_data, ex_class, timestamp)
                    timestamp += 1
    
                    # Offline - Evaluation
                    if (timestamp) % summarizer.time_gap == 0:
                        # FIMXE: Error in Sim 6 and thrsh 0.8
                        om = metrics.all_offline_metrics(summarizer._V,
                                                         summarizer._Vmm,
                                                         summarizer.summary())
                        max_memb = np.max(summarizer._Vmm, axis=0)
                        tot_memb = np.sum(summarizer._Vmm, axis=0)
                        purity = np.sum(max_memb / tot_memb) / len(max_memb)
                        print(f"Offline purity for {timestamp}: {purity}")
                        print(om)
   
    
                # Obtain al metrics and create the row
                all_metrics = metrics.all_online_metrics(summarizer.summary(),
                                                         chunksize)
                metrics_summary = ""
                for name, value in all_metrics.items():
                    metrics_summary += f"{name}: {round(value,3)}\n"
                metrics_summary = metrics_summary[:-1]
    
                row_metrics = list(all_metrics.values())
                row_timestamp = ["[" + str(timestamp) + " to " + str(timestamp + chunksize - 1) + "]"]
    
                new_row = pd.DataFrame([row_timestamp + row_metrics],
                                       columns=df.columns)
                df = pd.concat([df, new_row], ignore_index=True)
    
                
                # print("Total de Fmics = "+str(len(summarizer.summary())))
                for fmic in summarizer.summary():    
                    summary['x'].append(fmic.center[0])
                    summary['y'].append(fmic.center[1])
                    summary['radius'].append(fmic.radius * 100000)
                    summary['weight'].append(fmic.m)
    
            
            # Transforming FMiCs into dataframe
            for fmic in summarizer.summary():
                summary['x'].append(fmic.center[0])
                summary['y'].append(fmic.center[1])
                summary['radius'].append(fmic.radius * 100_000)
                summary['weight'].append(fmic.m)
                summary['class'].append(max(fmic.tags, key=fmic.tags.get))
                summary['color'].append(color[max(fmic.tags, key=fmic.tags.get)])
                
            if Figures:
                if not os.path.isdir(f"{output_path}/Img/"):
                    os.mkdir(f"{output_path}/Img/")

                fig = plt.figure()
                # Plot radius
                plt.scatter('x', 'y', s='radius', color='color',
                            data=summary, alpha=0.1)
                # Plot centroids
                plt.scatter('x', 'y', s=1, color='color', data=summary)
                # plt.legend(["color blue", "color green"], loc ="lower right")
                # plt.legend(["Purity"+str(summarizer.Purity()),"PartitionCoefficient"+str(summarizer.PartitionCoefficient()),"PartitionEntropy"+str(summarizer.PartitionEntropy()),"XieBeni"+str(summarizer.XieBeni()), "FukuyamaSugeno_1"+str(summarizer.FukuyamaSugeno_1()),"FukuyamaSugeno_2"+str(summarizer.FukuyamaSugeno_2())], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                # plt.figtext(.8, .8, "T = 4K")
                # side_text = plt.figtext(.91, .8, metrics_summary)
                fig.subplots_adjust(top=1.0)
                # plt.show()
                fig_name = (f"{output_path}/Img/[Chunk {timestamp - 1000} to {timestamp - 1})]"
                            f" Sim({simIDX})_Thresh({threshIDX}).png")
                fig.savefig(fig_name, # bbox_extra_artists=(side_text,),
                            bbox_inches='tight')
                plt.close()
    
            print("==== Approach ====")
            print("Similarity = ", simIDX)
            print("Threshold = ", threshIDX)
            # print("==== Summary ====")
            # print(summary)
            print("==== Metrics ====")
            print(summarizer.metrics)
            print("\n")
            print(df)
            print("------")
    
            tabRes.iloc[vecIndex, list(range(0, numChunks * 3, 3))] = df['XieBeni']
            tabRes.iloc[vecIndex, list(range(1, numChunks * 3, 3))] = df['pCoefficient']
            tabRes.iloc[vecIndex, list(range(2, numChunks * 3, 3))] = df['MPC']
            tabRes.iloc[vecIndex, -3] = df['XieBeni'].mean()
            tabRes.iloc[vecIndex, -2] = df['pCoefficient'].mean()
            tabRes.iloc[vecIndex, -1] = df['MPC'].mean()

            df = df[0:0]

    
        tabRes.to_excel("".join((output_path, f"Selected_thresholds_{dataset}_{start}_{end}_{max_fmics}_{chunksize}_{fname}.xlsx")))
    print("--- End of execution --- ")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    parser.add_argument('--dataset', type=str, default='Benchmark1_11000',
                          help='Dataset: Benchmark1_11000 (d) or RBF1_40000')

    parser.add_argument('--chunksize', type=int, default=1000,
                          help='Chunk size of the stream (d = 1000)')

    parser.add_argument('--min_fmics', type=int, default=5,
                          help='Minimun number of fmics (d = 5)')

    parser.add_argument('--max_fmics', type=int, default=100,
                          help='Maximun number of fmics (d = 100)')
    
    parser.add_argument('--start', type=int, default=0,
                          help='starting similarity measure index (d = 0)')

    parser.add_argument('--end', type=int, default=0,
                          help='ending similarity measure index  (d = 0 last)')

    parser.add_argument('--fname', type=str, default="experiment",
                          help='name to add to the file')
    
    args = parser.parse_args()

    dataset = args.dataset 
    chunksize = args.chunksize
    min_fmics = args.min_fmics
    max_fmics = args.max_fmics
    start = args.start
    end = args.end
    fname = args.fname
    
    experiment(dataset, chunksize, min_fmics, max_fmics, start, end, fname)
