# ACSC

C. Fahy, S. Yang and M. Gongora, "Ant Colony Stream Clustering: A Fast Density Clustering Algorithm for Dynamic Data Streams," in IEEE Transactions on Cybernetics, vol. 49, no. 6, pp. 2215-2228, June 2019, doi: 10.1109/TCYB.2018.2822552

https://ieeexplore.ieee.org/document/8357585

[CODE](ACSC/)


## Usage example:

```
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ACSC.ACSC import ACSC

nSamples = 3
sleepMax = 3
epsilon = 0.1
dataset = "dataset_paht.csv"
window_length = 1000

acsc = ACSC(nSamples, sleepMax, epsilon) 
print(f"ACSC for {dataset=}")

with pd.read_csv(dataset,
                    dtype=dtypes,
                    chunksize=window_length) as reader:
    timestamp = 0
    ARI = []
    for chunk in reader:
        print(f"Summarizing examples from {timestamp} to {timestamp + window_length - 1}")
        
        X = chunk.iloc[:,:-1].to_numpy()
        y = chunk.iloc[:,-1].fillna("-1").to_numpy()
        acsc.process_window(X)
        
        final_clusters = acsc.clusters
        print(f"{len(final_clusters)} Clusters found.")
        
        labels_ASCS = np.array([acsc.predict(x) for x in X])
        label_real = y
        label_real[label_real == np.nan] = -1
        
        ARI.append(adjusted_rand_score(label_real[-window_length:], labels_ASCS[-window_length:]))
        
        print(ARI[-1])       
        timestamp += window_length
    print(np.mean(ARI))
```
