#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:02:34 2023

@author: asier
"""

import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, f1_score
from src.functions.distance import EuclideanDistance


def all_online_metrics(fmics, chunksize):
    return {
        "Purity": Purity(fmics),
        "PartitionCoefficent": PartitionCoefficient(fmics, chunksize),
        "PartitionEntropy": PartitionEntropy(fmics, chunksize),
        "XieBeni": XieBeni(fmics, chunksize),
        "ModifiedPartitionCoefficent": ModifiedPartitionCoefficient(fmics,
                                                                    chunksize),
        "FukuyamaSugeno_1": FukuyamaSugeno_1(fmics),
        "FukuyamaSugeno_2": FukuyamaSugeno_2(fmics, chunksize)
        }


def offline_stats(summarizer, chunk):
    """
    Computes stats for the offline step given the summarizer structure and the last chunk of data

    Args:
        summarizer: dFuzzStream summarizer structure
        chunk: Pandas Data frame. Each column is an attribute and the last column should be the class

    Returns:
        ari, sil: Adjusted Rand Index, Silhouette score
    """
    fmicsc = [f.center.to_list() for f in summarizer.summary()]
    clusters = np.argmax(summarizer._Vmm, axis=0)
    clusters[np.max(summarizer._Vmm,
                    axis=0) < 0.5] = -1  # if the highest membership to a cluster is < 0.5 then it is an outlier
    point_fmic = []
    for _, point in chunk.iterrows():
        d_min = EuclideanDistance.distance(point[:2], fmicsc[0])
        id_min = 0
        for i, fm in enumerate(fmicsc):
            d = EuclideanDistance.distance(point[:2], fm)
            if d < d_min:
                d_min = d
                id_min = i
        point_fmic.append(clusters[id_min])
    y = chunk[chunk.columns[-1]].values

    y_h = np.array(point_fmic)
    # not_nans = np.where(y.astype(str) != 'nan')[0].astype(int)
    nans = np.where(y.astype(str) == 'nan')[0].astype(int)
    y[nans] = -1
    y = y.astype(int)
    # ari = adjusted_rand_score(y[not_nans], y_h[not_nans])
    ari = adjusted_rand_score(y, y_h)
    # f1 = f1_score(y, y_h)
    if len(np.unique(y_h))==1:
        sil = 0
    else:
        sil = silhouette_score(chunk[chunk.columns[:-1]].values, y_h)

    return ari, sil

def all_offline_metrics(cluster_centers, membership_matrix, fmics):
    classes = np.array([max(fm.tags) for fm in fmics])
    clusters = np.argmax(membership_matrix, axis=0)
    points = np.array([fm.center for fm in fmics])
    return {
        "Purity": offline_purity(membership_matrix),
        "ARI": ARI(classes, clusters),
        "FS": FS(points, cluster_centers, membership_matrix)
        }


def ARI(classes, clusters):
    '''
    Adjusted Rand Index

    Parameters
    ----------
    classes : Array
        Array with the class of each example.
    clusters : Array
        Array with the cluster asigned to each example.

    Returns
    -------
    ARI : Float
        Adjusted Rand Index.

    '''
    classes_n = np.unique(classes)
    clusters_n = np.unique(clusters)

    n = len(classes)
    a = [len(np.where(classes == cla)[0]) for cla in classes_n]
    b = [len(np.where(clusters == clu)[0]) for clu in clusters_n]
    nij = [[len(np.where((clusters == clu) & (classes == cla))[0])
            for clu in clusters_n]
           for cla in classes_n]
    n2 = np.math.comb(n, 2)
    Eai2 = np.sum([np.math.comb(ai, 2) for ai in a])
    Ebj2 = np.sum([np.math.comb(bj, 2) for bj in b])
    Eij2 = np.sum([[np.math.comb(n, 2) for n in row] for row in nij])
    ARI = (Eij2 - (Eai2 * Ebj2) / n2) / (0.5 * (Eai2 + Ebj2) - (Eai2 * Ebj2) / n2)

    return ARI


def offline_purity(memebership_matrix):
    '''
    Purity

    Parameters
    ----------
    memebership_matrix: Array
        Matrix with the membership of each example to each cluster

    Returns
    -------
    Purity : Float
        Purity.
    '''
    max_memb = np.max(memebership_matrix, axis=0)
    tot_memb = np.sum(memebership_matrix, axis=0)
    purity = np.sum(max_memb / tot_memb) / len(max_memb)
    return purity


def FS(x, c, mu, alpha=1):
    '''
    Fuzzy Silhouette

    Parameters
    ----------
    x : Array data points
        DESCRIPTION.
    c : Cluster's centroids
        DESCRIPTION.
    mu : Membership matrix
        DESCRIPTION.
    alpha : Fuzzy weighted coefficient
        Default 1

    Returns
    -------
    Float [0,1].

    '''
    w = [1 for _ in range(len(x))]
    return WFS(x, c, w, mu, alpha)


def WFS(x, c, w, mu, alpha=1):
    '''
    Weighted Fuzzy Silhouette

    Parameters
    ----------
    x : Array data points
        DESCRIPTION.
    c : Cluster's centroids
        DESCRIPTION.
    w : Weigths of examples
        DESCRIPTION.
    mu : Membership matrix
        DESCRIPTION.
    alpha : Fuzzy weighted coefficient
        Default 1

    Returns
    -------
    Float [0,1].

    '''
    n, s = x.shape
    nc = len(c)
    dist = np.zeros([n, nc])
    for i in range(nc):
        dist[:, i] = np.sqrt(np.sum((x - c[i, :])**2, axis=1).astype('float'))  # Euclidean
    labels = np.argmin(dist, axis=1)
    NC = [sum(labels == i) for i in range(nc)]  # points per cluster
    dm = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            dm[i][j] = np.sqrt(sum((x[i] - x[j])**2))
            dm[j][i] = dm[i][j]

    s = []
    for i in range(n):
        # aij
        aij = 0
        cluster = labels[i]
        same_cluster_idx = np.argwhere(labels == cluster)[:, 0]
        for j in same_cluster_idx:
            if i != j:
                aij += dm[i, j]
        aij = aij / (NC[cluster] - 1)

        # bij
        bij = []
        for p in range(nc):
            if p != cluster:
                bij.append(0)
                other_cluster_idx = np.argwhere(labels == p)[:, 0]
                for j in other_cluster_idx:
                    bij[-1] += dm[i, j]
                bij[-1] = bij[-1] / NC[p]
        bij = min(bij)

        si = (bij - aij) / max(aij, bij)
        s.append(si)

    # https://www.w3resource.com/python-exercises/numpy/advanced-numpy-exercise-11.php
    second_largest = np.partition(mu, -2, axis=0)
    muijp = second_largest[-1] - second_largest[-2]

    wfs = sum((muijp**alpha) * s * w) / sum((muijp**alpha) * w)
    return wfs


def Purity(fmics):
    partialpur = 0
    for idxFMIC, fmic in enumerate(fmics):
        majorityClass = max(fmic.tags.values())
        totalPoints = sum(fmic.tags.values())
        partialpur += majorityClass / totalPoints
    return (partialpur / len(fmics))


def PartitionCoefficient(fmics, chunksize):
    mSquare = 0
    for idxFMIC, fmic in enumerate(fmics):
        mSquare += fmic.mSquare

    return (1 / chunksize * mSquare)


def ModifiedPartitionCoefficient(fmics, chunksize):
    c = len(fmics)
    return 1 - ((c / (c - 1)) * (1 - PartitionCoefficient(fmics, chunksize)))


def PartitionEntropy(fmics, chunksize):
    mLog = 0
    for idxFMIC, fmic in enumerate(fmics):
        mLog += fmic.mLog

    return (-1 / chunksize * mLog)


def XieBeni(fmics, chunksize):
    sumaSSD = 0
    centroidList = np.ones((len(fmics), 2)) * 1000000
    menorDistancia = 1000000
    # storing the distances among all Fmics
    for idxFMIC, fmic in enumerate(fmics):
        sumaSSD += fmic.ssd
        centroidList[idxFMIC, :] = fmic.center

    MinDist = np.min(np.linalg.norm(centroidList, axis=1))

    return (1 / chunksize * sumaSSD) / MinDist


def FukuyamaSugeno_1(fmics):
    sumaSSD = 0
    centroidList = np.ones((len(fmics), 2))
    membershipList = np.ones(len(fmics))

    for idxFMIC, fmic in enumerate(fmics):
        sumaSSD += fmic.ssd
        centroidList[idxFMIC, :] = fmic.center
        membershipList[idxFMIC] = fmic.m

    V1 = np.sum(centroidList / len(fmics), axis=0)

    return sumaSSD - np.sum(membershipList * np.linalg.norm(centroidList - V1, axis=1))


def FukuyamaSugeno_2(fmics, chunksize):
    sumaSSD = 0
    sumaValues = 0
    centroidList = np.ones((len(fmics), 2))
    membershipList = np.ones(len(fmics))

    for idxFMIC, fmic in enumerate(fmics):
        sumaSSD += fmic.ssd
        centroidList[idxFMIC, :] = fmic.center
        membershipList[idxFMIC] = fmic.m
        sumaValues += 1 / chunksize * fmic.values

    V2 = sumaValues

    return sumaSSD - np.sum(membershipList * np.linalg.norm(centroidList - V2, axis=1))
