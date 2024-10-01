#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:10:33 2023

@author: asier
"""
import numpy as np
from src.functions import distance
from src.functions.metrics import WFS

def FCM(data, c, m=2, epsilon=0.001):
    '''
    Fuzzy c-means

    Parameters
    ----------
    data : dataframe
        x, y and weight columns
    c : int
        cluster number
    m : int, optional
         Fuzzy ... The default is 2.
    epsilon : float, optional
        Threshold for change in J. The default is 0.001.

    Returns
    -------
    V : Array of floats [2,c]  2 = x and y
        Array of centroids

    '''
    w = [1 for _ in range(len(data))]
    V, U = WFCM(data, w, c, m, epsilon)
    return V


def WFCM(data, weights, c, m=2, epsilon=0.001, it_max=300):
    '''
    Weighted Fuzzy c-means

    Parameters
    ----------
    data : dataframe
        x, y and weight columns
    weigths : Array
        weight of each sample
    c : int
        cluster number
    m : int, optional
         Fuzzy ... The default is 2.
    epsilon : float, optional
        Threshold for change in J. The default is 0.001.
    it_max: int, optional
        Maximum number of iterations

    Returns
    -------
    V : Array of floats [2,c]  2 = x and y
        Array of centroids
    U : Array of floata [n, c]
        Membership of each example to each cluster

    '''
    x = data
    n, s = len(x), len(x[0])
    # TODO: test V Initialization with the c points with higher weight
    c = min(c, len(x))  # It has happened to have less microclusters than c
    maxw_idx = np.argpartition(weights, -c)[-c:]
    V = data[maxw_idx]  # Centroid of clusters
    # V = np.zeros([c, s])  # Centroid of clusters
    # FIXME: I defaulted a seed to allow easier refactorization and repeatability
    rng = np.random.default_rng(seed=42)
    U = rng.random([c, n])
    # U = np.random.random([c, n])
    U = U/U.sum(axis=0, keepdims=True)
    D = np.zeros([c, n])
    J = 0
    newJ = 1
    it = 0

    while (abs(newJ-J) > epsilon) and (it<it_max):
        J = newJ
        # Step2: compute V
        for i in range(c):  # For each center
                for j in range(s): # for each attribute
                    V[i, j] = sum(weights[:] * (U[i, :]**2) * x[:, j])/sum(weights[:] * U[i, :]**2)  # * weigths[k]

        # Step3: compute D and U
        for i in range(c):
            for j in range(n):
                D[i, j] = distance.EuclideanDistance.distance(V[i], x[j])

        for i in range(c):
            for j in range(n):
                sum2 = 0
                for k in range(c):
                    sum2 += (D[i, j]/D[k, j])**(2/(m-1))
                U[i, j] = 1/sum2
        # U = U/U.sum(axis=1, keepdims=True)

        # Step4: compute J
        newJ = np.sum((U**m)*(D**2))
        it += 1
    return V, U



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.array([[1, 3], [2, 5], [6, 8], [7, 9]])
    w = [1, 1, 1, 1]
    v, mu = WFCM(x, w, 2, m=2, epsilon=0.001)
    plt.scatter(x[:, 0], x[:, 1], c='tab:blue', label='Points')
    plt.scatter(v[:, 0], v[:, 1], c='tab:orange', label='Centers')
    plt.show()
    wfs = WFS(x, v, w, mu)
    print(wfs)
