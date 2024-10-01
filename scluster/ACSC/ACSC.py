from random import randrange, random
from functools import partial
import numpy as np
from scluster.ACSC import microcluster

def find_clusters(data):
    clusters = []
    cluster_similarity = []
    for x in data:
        if len(clusters) != 0:
            bc = find_best_cluster()
            bc.add_point(x)
            update_cluster_similarity()
        elif len(clusters) != 0 and no_suitable_cluster():
            nc = microcluster()
            nc.add_point(x)
            update_cluster_similarity()

    return clusters, cluster_similarity


class ACSC():

    def __init__(self, nSamples:int, sleepMax:int = 3, epsilon:float=1.) -> None:
        self.epsilon = epsilon
        self.nSamples = nSamples
        self.clusters = []
        self.centers = []
        self.similarities = np.array([])
        self.sleepMax = sleepMax

    def suitability(self, p:microcluster, c:list[microcluster]) -> float: 
        # c contains all the points of the cluster
        n = len(c)
        mn_samp_n = min(self.nSamples, n)
        dist_sum = 0
        for j in range(mn_samp_n):
            dist_sum += np.sqrt(np.sum((p.c-c[j].c)**2))
            
        # dist_sum = sum(map(lambda cj: np.sqrt(np.sum((p-cj)**2)), c[:mn_samp_n]))/mn_samp_n # it seeems to work ok
        return dist_sum / mn_samp_n
    
    def similarity(self, a:list[microcluster], b:list[microcluster]) -> float:
        n = len(a)
        sim_ab = 0
        for ai in a:
            sim_ab +=  self.suitability(ai, b)
        # sum(map(lambda ai: self.suitability(ai, b), a))/n
        return sim_ab/n

        

    def add_point(self, x:np.array) -> None:
        if len(self.clusters) == 0:
            self.clusters.append(x)
        else:
            suitabilities = map(lambda c: self.suitability(x, c), self.clusters)
            best_cluster = argmin(suitabilities)
            best_suitability = suitabilities[best_cluster]
            mc = microcluster()
            mc.add_point(x)
            if best_suitability <= self.epsilon:
                self.clusters[best_cluster].append(mc)
            else:
                self.clusters.append(mc)

    def update_cluster_similarity(self, cluster_idx:int) -> None:
        # It seems to repeat to many comparison...
        for i in range(len(self.clusters)):
            sim_idx_i = self.similarity(self.clusters[cluster_idx], self.clusters[i])
            self.similarities[cluster_idx, i] = sim_idx_i
            sim_i_idx = self.similarity(self.clusters[i], self.clusters[cluster_idx])
            self.similarities[i, cluster_idx] = sim_i_idx

    
    def find_clusters(self, data:np.array) -> None:
        for x in data:
            new_mc = microcluster.microcluster()
            new_mc.add_point(x)
            suitable_cluster = False
            if len(self.clusters) != 0:
                suitabilities = np.array(list(map(lambda c: self.suitability(new_mc, c), self.clusters)))
                # np.apply_along_axis(lambda c: self.suitability(x,c), 0, self.clusters) # Does not work...
                best_cluster_idx = np.argmin(suitabilities)
                best_suitability = suitabilities[best_cluster_idx]
                if best_suitability <= self.epsilon:
                    self.clusters[best_cluster_idx].append(new_mc)
                    suitable_cluster = self.clusters[best_cluster_idx]
                    self.update_cluster_similarity(best_cluster_idx)  # Similarity...
            if len(self.clusters) == 0 or not suitable_cluster:
                self.clusters.append([new_mc])
                n_clusters = len(self.clusters)
                self.similarities.resize((n_clusters,n_clusters), refcheck=False)
                self.update_cluster_similarity(n_clusters - 1) # (6) similarity

    def create_merge_microclusters(self) -> list[list[microcluster]]:
        all_mclusters = []
        for cluster in self.clusters:
            mclusters = []
            for point in cluster:
                mclusters.append(point)
            for i in range(len(mclusters)-1,-1,-1):
                m_merge = False
                for j in range(len(mclusters)-2,-1,-1):
                    m_merge = microcluster.merge_microclusters(mclusters[i],mclusters[j],self.epsilon)
                    if m_merge:
                        del mclusters[i]
                        del mclusters[j]
                        mclusters.append(m_merge)
                        break # I don't follow with the loop because i is not longer valid
                if m_merge:
                    continue  # if merged i is not valid and I start with i-1 which is the new merged microcluster

            all_mclusters.append(mclusters)
        
        return all_mclusters


# “Sorting ants” are created and one is assigned to each cluster. 
# Each sorting ant is native to its own cluster. 

# Sorting ants probabilistically decide to pick-up a micro-cluster from their cluster. 
# A micro-cluster m is chosen at random from cluster k and is iteratively compared with nSamples micro-clusters in the same cluster. 
# The Euclidean distance from the center of m to each of the selected micro-clusters is calculated and if both are density-reachable (4), 
# then a reachable count is incremented. 
# The probability of a pick-up is calculated as follows:  Ppick = 1 - reachable / nSamples (8)
# It is important to note that if the number of micro-clusters n in cluster c is fewer than nSamples, then only n comparisons are made. 
# However, Ppick is still calculated using nSamples. This ensures a higher probability of a pick-up in clusters containing fewer micro-clusters. 
# This leads to the dissolution of smaller clusters and their incorporation into larger, similar clusters. 

# If a micro-cluster is successfully picked-up, the Boolean variable carrying is true and the sorting ant moves to a neighboring cluster 
# and attempts to drop it. Sorting ants move to the most similar cluster (using the similarity information from the first step) 
# ensuring that they do not attempt to drop micro-clusters in clusters that are dissimilar to their own. 

# A sorting ant attempts to drop its micro-cluster in the new cluster based on the inverse of (8). 

# If the dropping operation is successful, the micro-cluster is moved to the new cluster; 
# otherwise, the micro-cluster remains in its original cluster. 

# The ant returns to its native cluster and updates the similarity information between the two clusters with the latest suitability score [see (5)]. 
# Each sorting ant continues to attempt sorting until either the cluster is empty (all of its contents have been moved to another cluster) 
# or the sorting ant is “asleep.” 

# Each sorting ant has a counter and if a pick-and-drop operation is unsuccessful, either picking or dropping, this counter is incremented. 
# When the counter reaches sleepMax, then the cluster is considered to be sorted and a Boolean counter sleeping is true. 
# The counter is reset to zero after a successful operation or if a new micro-cluster is placed in the cluster by a foreign sorting ant. 
# When all ants are sleeping, the stop condition is met.


    def sort_clusters(self):
        clusters_m = self.create_merge_microclusters()
        ants = np.zeros(len(clusters_m))    # to store ant counter
        while sum(ants) < len(clusters_m)*self.sleepMax:  # all of them are sleeping
            for i  in range(len(ants)):   # for each ant (by id)
                cluster = clusters_m[i]   # select its cluster
                if ants[i] < self.sleepMax:   # if it is not sleeping
                    mc_idx = np.random.choice([x for x in range(len(cluster))], 1)[0]
                    mc = cluster[mc_idx]  # select a microcluster at random
                    if len(cluster) > 1: # only one microcluster so reachability = 0, ppick = 1
                        n_select = min(self.nSamples, len(cluster))
                        choices = np.random.choice([x for x in range(len(cluster)) if x != mc_idx], n_select)  # select nSamples/n microclusters, excluding itself. Repetitions are possible 
                        distances = np.array(list(map(lambda x: np.sqrt(np.sum((cluster[x].c-mc.c)**2)), choices)))
                        reachable = sum(distances <= self.epsilon)
                        ppick = 1 - reachable/self.nSamples
                    else:
                        ppick = 1
                    if ppick > random(): 
                        closest_idx = np.argmin(self.similarities[i])
                        closest_cluster = clusters_m[closest_idx]

                        choices_dest = np.random.choice([x for x in range(len(closest_cluster))], self.nSamples)  # select nSamples microclusters. Repetitions are possible 
                        distances_dest = np.array(list(map(lambda x: np.sqrt(np.sum((cluster[x].c-mc.c)**2)), choices_dest)))
                        reachable_dest = sum(distances_dest <= self.epsilon)
                        pdrop = 1 - reachable_dest/self.nSamples
                        pdrop = 1 - self.nSamples/reachable_dest
                        pdrop = reachable_dest/self.nSamples

                        if  1 - ppick > random(): 
                            ants[i] = O
                            ants[closest_idx] = O
                            cluster.remove(mc)
                            closest_cluster.append(mc)
                            self.update_cluster_similarity(i)
                            self.update_cluster_similarity(closest_idx)
                        else:  # Not moved
                            ants[i] +=1
                    else:
                        ants[i] +=1
        
        self.clusters = clusters_m

    def compute_centers(self) -> None:
        centers = []
        for c in self.clusters:
            for mc in c:
                N = 0
                Cent = np.zeros(c[0].c.shape)
                for mc in c:
                    N += mc.N
                    Cent += mc.c * N
            centers.append(Cent/N)
        self.centers = centers

    def predict(self, x:np.array) -> int:
        d = [np.sqrt(np.sum((x-c)**2)) for c in self.centers] 
        cluster_idx = np.argmin(d)
        return cluster_idx

    
    def process_window(self, data:np.array) -> None:
        self.find_clusters(data)
        self.sort_clusters()
        self.compute_centers()
