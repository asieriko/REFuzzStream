# TSF-DBSCAN

A. Bechini, F. Marcelloni and A. Renda, "TSF-DBSCAN: A Novel Fuzzy Density-Based Approach for Clustering Unbounded Data Streams," in IEEE Transactions on Fuzzy Systems, vol. 30, no. 3, pp. 623-637, March 2022, doi: 10.1109/TFUZZ.2020.3042645

https://ieeexplore.ieee.org/abstract/document/9281371

[TSF_DBSCAN.py](TSF_DBSCAN.py)


## Usage example:

```
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from TSF_DBSCAN import TSF_DBSCAN, p_object

run_tsf_dbscan(dataset, outputPath, dtypes, chunksize, emin, emax, alpha, omega, wmin, break_n):

emin = 0.1
emax = 0.2
alpha = 0.5
omega = 0.3
wmin = 2
chunksize = 1000

tsf = TSF_DBSCAN(emin, emax, alpha, omega, wmin, chunksize) 

with pd.read_csv(dataset,
                    dtype=dtypes,
                    chunksize=chunksize) as reader:
    timestamp = 0
    ARI = []
    for chunk in reader:
        print(f"Summarizing examples from {timestamp} to {timestamp + chunksize - 1}")
        
        for index, example in chunk.iterrows():
            point = p_object(example[0:-1].tolist(), t=timestamp)
            tsf.tsfdbscan(point)
            timestamp += 1
            
        print(f"{len(tsf.clusters)} Clusters found. ({index})")

        results = np.array([x.x + list(x.get_max_cluster_membership()) for x in tsf.plist])
        X = results[:, :-2]
        labels_tsf = results[:, -2]
        M = results[:, -1]

        label_real = chunk.iloc[:,-1].copy()
        label_real[label_real.isnull()] = -1
        ARI.append(adjusted_rand_score(label_real.iloc[-chunksize:].astype('int'), labels_tsf[-chunksize:]))
        print(ARI[-1])
                
    print(np.mean(ARI))
```

