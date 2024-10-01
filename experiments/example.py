#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 09:27:46 2023

@author: asier
"""
import os, sys
from pathlib import Path
sys.path.append(os.path.abspath("."))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.RE_dFuzzStream import REdFuzzStreamSummarizer
from src.functions.merge import AllMergers
from src.functions.distance import EuclideanDistance
from src.functions.membership import FuzzyCMeansMembership
from src.functions import metrics

sm = 1
min_fmics = 5
max_fmics = 50
thresh = 0.8
chunksize = 100
color = {'1': 'Red', '2': 'Blue', '3': 'Green', '4': 'pink', 'nan': 'Gray'}

# Folder to store the produced images
if not os.path.isdir("./Img/"):
    os.mkdir("./Img/")

datasetPath = Path.cwd() / "datasets"/ "DS1.csv"

df = pd.DataFrame(columns = ['Chunk', 'Purity', 'pCoefficient', 'pEntropy', 'XieBeni','MPC','FukuyamaSugeno_1','FukuyamaSugeno_2'])
summarizer = REdFuzzStreamSummarizer(
    distance_function=EuclideanDistance.distance,
    merge_threshold = thresh,
    merge_function=AllMergers[sm](sm, thresh, max_fmics),
    membership_function=FuzzyCMeansMembership.memberships,
    chunksize = chunksize,
    n_macro_clusters=4,
    time_gap=100,
)

summary = {'x': [], 'y': [], 'radius' : [], 'color': [], 'weight': [], 'class': []}
timestamp = 0

# Read files in chunks
with pd.read_csv(datasetPath,
                dtype={"X1": float, "X2": float, "class": str},
                chunksize=chunksize) as reader:
    for chunk in reader:
        print(f"Summarizing examples from {timestamp} to {timestamp + 999} -> sim {sm} and thrsh {thresh}")
        for index, example in chunk.iterrows():
            # Summarizing example
            summarizer.summarize(example[0:-1], example[-1], timestamp)
            timestamp += 1
        summarizer.offline()


        # TODO: Obtain al metrics and create the row
        all_metrics = metrics.all_online_metrics(summarizer.summary(), chunksize)
        metrics_summary = ""
        for name, value in all_metrics.items():
            metrics_summary += f"{name}: {round(value,3)}\n"
        metrics_summary = metrics_summary[:-1]

        row_metrics = list(all_metrics.values())
        row_timestamp = ["["+str(timestamp)+" to "+str(timestamp + 999)+"]"]

        new_row = pd.DataFrame([row_timestamp + row_metrics],
                               columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)

        for fmic in summarizer.summary():

            summary['x'].append(fmic.center[0])
            summary['y'].append(fmic.center[1])
            summary['radius'].append(fmic.radius * 1000)
            summary['color'].append(color[max(fmic.tags, key=fmic.tags.get)])
            summary['weight'].append(fmic.m)
            summary['class'].append(max(fmic.tags, key=fmic.tags.get))


        fig = plt.figure()
        # Plot radius
        plt.scatter('x', 'y', s='radius', color='color',
                    data=summary, alpha=0.1)
        # Plot centroids
        plt.scatter('x', 'y', s=1, color='color', data=summary)
        # plt.legend(["color blue", "color green"], loc ="lower right")
        # plt.legend(["Purity"+str(summarizer.Purity()),"PartitionCoefficient"+str(summarizer.PartitionCoefficient()),"PartitionEntropy"+str(summarizer.PartitionEntropy()),"XieBeni"+str(summarizer.XieBeni()), "FukuyamaSugeno_1"+str(summarizer.FukuyamaSugeno_1()),"FukuyamaSugeno_2"+str(summarizer.FukuyamaSugeno_2())], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.figtext(.8, .8, "T = 4K")

        side_text = plt.figtext(.91, .8, metrics_summary)
        fig.subplots_adjust(top=1.0)
        # plt.show()
        fig.savefig("./Img/Example_[Chunk "+str(timestamp - 1000)+" to "+str(timestamp - 1)+"] Sim("+str(sm)+")_Thresh("+str(thresh)+").png", bbox_extra_artists=(side_text,), bbox_inches='tight')
        plt.close()
        plt.close()


    # Transforming FMiCs into dataframe
    for fmic in summarizer.summary():
        summary['x'].append(fmic.center[0])
        summary['y'].append(fmic.center[1])
        summary['radius'].append(fmic.radius * 1000)
        summary['color'].append(color[max(fmic.tags, key=fmic.tags.get)])
        summary['weight'].append(fmic.m)
        summary['class'].append(max(fmic.tags, key=fmic.tags.get))

    print("==== Approach ====")
    print("Similarity = ", sm)
    print("Threshold = ", thresh)
    # print("==== Summary ====")
    # print(summary)
    # It is a big dict
    print("==== Metrics ====")
    print(summarizer.metrics)
    print("\n")
    print(df)
    print("------")

    df = df[0:0]

    print("Final clusters:")
    if timestamp % 100 == 0:
        final_clusters, mu = summarizer.final_clustering()
        for fc in final_clusters:
            print(fc)


print("--- End of execution --- ")
