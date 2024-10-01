# EvolveCluster

Run EvolveCluster.py to run all experiments.

The DELMH dataset cannot be shared here. The interested party, who wants to run those experiments, has to gain approval to use the dataset. Look here: https://www.datafirst.uct.ac.za/dataportal/index.php/catalog/759

If you intend to use this work, please cite the following paper:

Nordahl, C., Boeva, V., Grahn, H., Persson Netz, M. EvolveCluster: an evolutionary clustering algorithm for streaming data. Evolving Systems (2021). https://doi.org/10.1007/s12530-021-09408-y

Code by the authors: https://github.com/christiannordahl/EvolveCluster

In this folder is a copy of the code


## Usage example:
```

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from scluster.EvolveCluster import EvolveCluster
from scluster.EvolveCluster.functions import calculate_distances

dataset = "dataset_paht.csv"
chunksize = 1000
tau = 0.1

first = True
with pd.read_csv(dataset,
                    dtype=dtypes,
                    chunksize=chunksize) as reader:
    timestamp = 0
    ARI = []
    for chunk in reader:
        print(f"Summarizing examples from {timestamp} to {timestamp + chunksize - 1}")
        
        if first:
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
            
        print(f"{len(ec.clusters)} Clusters found.")

        
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
        print(ARI[-1])
        
        timestamp += len(chunk)

    print(np.mean(ARI))
```
