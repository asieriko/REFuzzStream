#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:40:51 2023

@author: asier
"""
import pandas as pd


class dataset:

    def __init__(
            self,
            source="",
            chunksize=1,
            numChunks=1,
            nLabels=None
            ):
        self.source = source
        self.chunksize = chunksize
        self.numChunks = numChunks
        self.nLabels = nLabels

    def get_chunk(self):
        with pd.read_csv(self.source,
                         dtype={"X1": float, "X2": float, "class": str},
                         chunksize=self.chunksize) as reader:
            for chunk in reader:
                yield chunk


class Benchmark1_11000(dataset):

    def __init__(self):
        datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv"
        super().__init__(datasetPath, 1000, 11, 3)


class RBF1_40000(dataset):

    def __init__(self):
        datasetPath = "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv"
        super().__init__(datasetPath, 1000, 40, 4)


if __name__ == "__main__":
    nd1 = Benchmark1_11000()
    for c in nd1.get_chunk():
        print(c.info())

    nd2 = RBF1_40000()
    for c in nd2.get_chunk():
        print(c.info())
