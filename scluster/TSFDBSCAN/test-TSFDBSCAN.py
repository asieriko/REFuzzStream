#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:56:54 2023

@author: asier
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from TSF_DBSCAN import TSF_DBSCAN, p_object


def plot(X, C, M, outliers=[], ranges=None, title="title"):
    if any(outliers):
        Xo = X[outliers]

        notoutliers = np.logical_not(outliers)
        Xc = X[notoutliers]
        Mc = M[notoutliers]
        Cc = C[notoutliers]
    else:
        Xc = X
        Mc = M
        Cc = C

    uC = [c for c in np.unique(Cc)]

    for i in range(len(np.unique(uC))):
        Cc[np.where(Cc == uC[i])] = i

    if ranges is not None:
        x1_min, x2_min = ranges[0]
        x1_max, x2_max = ranges[1]
    else:
        x1_min, x2_min = [round(x) for x in np.min(X[:, :-1], axis=0)]
        x1_max, x2_max = [round(x) for x in np.max(X[:, :-1], axis=0)]

    cmap = plt.get_cmap("tab20b", int(np.max(C)) - int(np.min(C)) + 1)
    plt.figure()
    if any(outliers):
        plt.scatter(Xo[:, 0], Xo[:, 1], marker='x')
    if len(Xc > 0):
        plt.scatter(Xc[:, 0], Xc[:, 1], c=Cc, cmap=cmap, alpha=Mc)  # alpha=M  # for fuzzy borders
    plt.xlim(x1_min - 1, x1_max + 1)
    plt.ylim(x2_min - 1, x2_max + 1)
    plt.colorbar()
    plt.title(title)

    # TODO: Save?


def boxplot(data, names, title):
    plt.figure()
    plt.boxplot(data,
                vert=True,  # vertical box alignment
                patch_artist=True,  # fill with color
                labels=names)  # will be used to label x-ticks
    plt.ylim(0, 1)
    plt.title(title)
    plt.show()

    plt.figure()
    plt.plot(data[0])
    plt.plot(data[1])
    plt.ylim(0, 1)
    plt.title(names)
    plt.show()


def test_tsf_scan():
    currentPath = Path.cwd()
    dataset = "SamplesFile_b_4C2D800Linear.csv"
    filePath = currentPath.parent / "datasets" / dataset
    # To test the paramters for a given dataset
    data = np.loadtxt(filePath, delimiter=",")
    y = data[:, -1]
    # math.log(0.4,2)/0.0066 ~ -200.3 0.4 Ow, 0.0066 alpha, 200 toffline
    emin_l = np.arange(0.05, 0.41, 0.05)
    emin_l = np.arange(0.4, 1.01, 0.05)
    emax_m = np.arange(1, 11)
    # tsf = TSF_DBSCAN(1.7, 4, 0.0132, 0.4, 4, 100)
    print("emin, emax, t, clusters, ari, silhouette")
    for emin in emin_l:
        for emax in emax_m:
            tsf = TSF_DBSCAN(emin, emax * emin, 0.0132, 0.4, 4, 100)
            for i, p in enumerate(data):
                point = p_object(p[:-1].tolist(), t=i)
                tsf.tsfdbscan(point)
            # print(len(tsf.clusters), " Clusters found")
                if (i + 1) % 100 == 0:

                    results = np.array([x.x + list(x.get_max_cluster_membership()) for x in tsf.plist])
                    X = results[:, :-2]
                    C = results[:, -2]
                    M = results[:, -1]

                    outliers = (C == -1)

                    x1_min, x2_min = [round(x) for x in np.min(data[:, :-1], axis=0)]
                    x1_max, x2_max = [round(x) for x in np.max(data[:, :-1], axis=0)]
                    plot(X, C, M, outliers,
                         ranges=[[x1_min, x2_min], [x1_max, x2_max]],
                         title=f"TSF-DBSCAN {(i+1)*100}")

                    # plt.savefig(f"TSFDBSCAN/TSF-{emin}-{emin*emax:.2f}-{i + 1}.png")

                    yj = y[i - len(X):i]

                    try:
                        print(f"{emin}, {emax * emin:.2f}, {i + 1}, {len(tsf.clusters)}," +
                              f"{adjusted_rand_score(yj, C)}, {'-' if len(tsf.clusters) == 0 else silhouette_score(X, C)}")
                    except:
                        pass  # SIL gives error with nan


def main():
    currentPath = Path.cwd()
    dataset = "SamplesFile_b_4C2D800Linear.csv"
    dataset = "banana.csv"
    dataset = "covertype2.csv"
    filePath = currentPath / "datasets" / dataset
    T = 1000  # 800 banana, 100 DS1, 1000 covertype
    R = 581  # 6 banana, 8 DS1,

    data = np.loadtxt(filePath, delimiter=",")
    if dataset == "banana.csv":
        np.random.shuffle(data)  # For the banana dataset

    m = np.mean(data[:, :-1], axis=0)
    s = np.std(data[:, :-1], axis=0)

    data[:, :-1] = (data[:, :-1] - m) / s

    y = data[:, -1]
    # math.log(0.4,2)/0.0066 ~ -200.3 0.4 Ow, 0.0066 alpha, 200 toffline
    # tsf = TSF_DBSCAN(0.25, 1.25, 0.015, 0.3, 3.5, T)  # DS1
    # tsf = TSF_DBSCAN(0.30, 1.50, 0.0015, 0.3, 16, T)  # Banana
    tsf = TSF_DBSCAN(1., 1.50, 0.0015, 0.3, 1, T)  # Banana
    ARI = []
    SIL = []
    for j in range(R):
        for i, p in enumerate(data[:(j + 1) * T]):
            point = p_object(p[:-1].tolist(), t=i)
            tsf.tsfdbscan(point)
        print(f"{len(tsf.clusters)} Clusters found. ({j}/{R})")

        results = np.array([x.x + list(x.get_max_cluster_membership()) for x in tsf.plist])
        X = results[:, :-2]
        C = results[:, -2]
        M = results[:, -1]

        if len(X[0]) == 2:  # Plot only 2D datasets
            outliers = (C == -1)

            x1_min, x2_min = [round(x) for x in np.min(data[:, :-1], axis=0)]
            x1_max, x2_max = [round(x) for x in np.max(data[:, :-1], axis=0)]
            plot(X, C, M, outliers,
                 ranges=[[x1_min, x2_min], [x1_max, x2_max]],
                 title=f"TSF-DBSCAN {(j+1)*T}")

        ARI.append(adjusted_rand_score(y[j * T:(j + 1) * T], C[-T:]))
        if len(X) - 1 >= len(np.unique(C)) > 1:
            SIL.append(silhouette_score(X, C))
        else:
            SIL.append(np.nan)
        print(ARI[-1], SIL[-1])
    print(np.mean(ARI))
    print(np.mean(SIL))

    boxplot([ARI, SIL], ["ARI", "SIL"], title="ARI and SIL")


if __name__ == "__main__":
    # test_tsf_scan()
    main()
