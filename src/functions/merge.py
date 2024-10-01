from src.fmic import FMiC
from src.functions.distance import EuclideanDistance
import numpy as np


class AbstractFuzzyDissimilarityMerger:

    def __init__(self, sm, threshold, max_fmics, dims=3):
        self.similMatrix = np.zeros((max_fmics, max_fmics, dims))
        self.max_fmics = max_fmics
        self.sm = sm
        self.threshold = threshold
        self.dims = dims

    def delete(self, i):
        self.similMatrix = np.delete(self.similMatrix, i, axis=0)
        self.similMatrix = np.delete(self.similMatrix, i, axis=1)
        self._expand_matrix()

    def _expand_matrix(self):
        # Theoretically adding the rows and columns is not neccesary,
        # because there is one less MiC, but as we keep always the matrix
        # at its maximun size, and we don't add anything when a new mic
        # is created, this is the place to update
        n = self.max_fmics - len(self.similMatrix)
        temp = np.zeros((self.max_fmics, self.max_fmics, self.dims))
        temp[:-n, :-n] = self.similMatrix
        self.similMatrix = temp

    def _merge_matrix(self, i, j):
        pass

    def _similarity(self, fmics, memberships, i, j):
        pass

    def _update(self, fmics, memberships):
        fmics_to_merge = []
        for i in range(0, len(fmics) - 1):
            for j in range(i + 1, len(fmics)):
                similarity = self._similarity(fmics, memberships, i, j)  # FIXME

                if similarity >= self.threshold:
                    fmics_to_merge.append([i, j, similarity])
        return fmics_to_merge

    def merge(self, fmics, memberships):
        fmics_to_merge = self._update(fmics, memberships)
        # Sort by most similar
        fmics_to_merge.sort(reverse=True, key=lambda k: k[2])
        merged_fmics_idx = []
        idx_to_delete = []
        for (i, j, _) in fmics_to_merge:
            if i not in merged_fmics_idx and j not in merged_fmics_idx:
                fmics[i] = FMiC.merge(fmics[i], fmics[j])
                self._merge_matrix(i, j)
                idx_to_delete.append(j)
                merged_fmics_idx.extend([i, j])

        if len(idx_to_delete) > 0:
            idx_to_delete.sort(reverse=True)
            for idx in idx_to_delete:
                fmics.pop(idx)
                self.delete(idx)  # Expands matrix after deleting row/column

        return fmics


class FuzzyEuclideanMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics, dims=0)

    def _similarity(self, fmics, memberships, i, j):
        dissimilarity = EuclideanDistance.distance(fmics[i].center, fmics[j].center)
        sum_of_radius = fmics[i].radius + fmics[j].radius
        if dissimilarity != 0:
            similarity = sum_of_radius / dissimilarity
        else:
            # Highest value possible
            similarity = 1.7976931348623157e+308
        return similarity

    def delete(self, i):
        # No partial data stored, so no delete need
        pass


class FuzzyMinMaxMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _merge_matrix(self, i, j):
        # Minimum
        self.similMatrix[:i, i, 0] = (
            self.similMatrix[:i, i, 0] +
            self.similMatrix[:i, j, 0])
        self.similMatrix[i, i + 1:j, 0] = (
            self.similMatrix[i, i + 1:j, 0] +
            self.similMatrix[i + 1:j, j, 0])
        self.similMatrix[i, j + 1:, 0] = (
            self.similMatrix[i, j + 1:, 0] +
            self.similMatrix[j, j + 1:, 0])
        # Maximum
        self.similMatrix[:i, i, 1] = (
            self.similMatrix[:i, i, 1] +
            self.similMatrix[:i, j, 1])
        self.similMatrix[i, i + 1:j, 1] = (
            self.similMatrix[i, i + 1:j, 1] +
            self.similMatrix[i + 1:j, j, 1])
        self.similMatrix[i, j + 1:, 1] = (
            self.similMatrix[i, j + 1:, 1] +
            self.similMatrix[j, j + 1:, 1])

    def _similarity(self, fmics, memberships, i, j):
        self.similMatrix[i, j, 0] += np.minimum(memberships[i], memberships[j])
        self.similMatrix[i, j, 1] += np.maximum(memberships[i], memberships[j])
        similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]
        return similarity


class FuzzyREFMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _merge_matrix(self, i, j):
        # ref
        self.similMatrix[:i, i, 0] = (
            self.similMatrix[:i, i, 0] +
            self.similMatrix[:i, j, 0])
        self.similMatrix[i, i + 1:j, 0] = (
            self.similMatrix[i, i + 1:j, 0] +
            self.similMatrix[i + 1:j, j, 0])
        self.similMatrix[i, j + 1:, 0] = (
            self.similMatrix[i, j + 1:, 0] +
            self.similMatrix[j, j + 1:, 0])
        # Cardinality
        self.similMatrix[:i, i, 1] = (
            self.similMatrix[:i, i, 1] +
            self.similMatrix[:i, j, 1])
        self.similMatrix[i, i + 1:j, 1] = (
            self.similMatrix[i, i + 1:j, 1] +
            self.similMatrix[i + 1:j, j, 1])
        self.similMatrix[i, j + 1:, 1] = (
            self.similMatrix[i, j + 1:, 1] +
            self.similMatrix[j, j + 1:, 1])

    def _similarity(self, fmics, memberships, i, j):
        t = 2
        self.similMatrix[i, j, 0] += np.power(1 - np.absolute(memberships[i] - memberships[j]), 1/t)
        self.similMatrix[i, j, 1] += 1
        similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]
        return similarity


# Probabilistic Sum
class FuzzyAbstractPSMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _merge_matrix(self, i, j):
        # Initialization
        self.similMatrix[:i, i, 2] = np.maximum(
            self.similMatrix[:i, i, 2],
            self.similMatrix[:i, j, 2])
        self.similMatrix[i, i + 1:j, 2] = np.maximum(
            self.similMatrix[i, i + 1:j, 2],
            self.similMatrix[i + 1:j, j, 2])
        self.similMatrix[i, j + 1:, 2] = np.maximum(
            self.similMatrix[i, j + 1:, 2],
            self.similMatrix[j, j + 1:, 2])
        # Product
        self.similMatrix[:i, i, 1] = (
            self.similMatrix[:i, i, 1] *
            self.similMatrix[:i, j, 1])
        self.similMatrix[i, i + 1:j, 1] = (
            self.similMatrix[i, i + 1:j, 1] *
            self.similMatrix[i + 1:j, j, 1])
        self.similMatrix[i, j + 1:, 1] = (
            self.similMatrix[i, j + 1:, 1] *
            self.similMatrix[j, j + 1:, 1])
        # Similarity
        self.similMatrix[:i, i, 0] = 1 - self.similMatrix[:i, i, 0]
        self.similMatrix[i, i + 1:j, 0] = 1 - self.similMatrix[i, i + 1:j, 0]
        self.similMatrix[i, j + 1:, 0] = 1 - self.similMatrix[i, j + 1:, 0]

    def _func(self, membership_i, membership_j):
        pass

    def _similarity(self, fmics, memberships, i, j):
        result = self._func(memberships[i], memberships[j])
        if self.similMatrix[i, j, 2] == 0:
            self.similMatrix[i, j, 1] = (1 - result)
            self.similMatrix[i, j, 2] = 1
        else:
            self.similMatrix[i, j, 1] *= (1 - result)
        self.similMatrix[i, j, 0] = 1 - self.similMatrix[i, j, 1]
        similarity = self.similMatrix[i, j, 0]
        return similarity


class FuzzyPSProdMerger(FuzzyAbstractPSMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        prod = membership_i * membership_j
        return prod


class FuzzyPSMinMerger(FuzzyAbstractPSMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        minm = np.minimum(membership_i, membership_j)
        return minm


class FuzzyPSGMMerger(FuzzyAbstractPSMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        GM = np.sqrt(membership_i * membership_j)
        return GM


class FuzzyPSOBMerger(FuzzyAbstractPSMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        OB = np.sqrt((membership_i * membership_j) * np.minimum(membership_i, membership_j))
        return OB


class FuzzyPSODivMerger(FuzzyAbstractPSMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        ODiv = (membership_i * membership_j + np.minimum(membership_i, membership_j)) / 2
        return ODiv


# Maximum
class FuzzyAbstractMaxMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _merge_matrix(self, i, j):
        self.similMatrix[:i, i, 0] = np.maximum(
            self.similMatrix[:i, i, 0],
            self.similMatrix[:i, j, 0])
        self.similMatrix[i, i + 1:j, 0] = np.maximum(
            self.similMatrix[i, i + 1:j, 0],
            self.similMatrix[i + 1:j, j, 0])
        self.similMatrix[i, j + 1:, 0] = np.maximum(
            self.similMatrix[i, j + 1:, 0],
            self.similMatrix[j, j + 1:, 0])

    def _func(self, membership_i, membership_j):
        pass

    def _similarity(self, fmics, memberships, i, j):
        result = self._func(memberships[i], memberships[j])
        self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], result)
        similarity = self.similMatrix[i, j, 0]
        return similarity


class FuzzyMaxProdMerger(FuzzyAbstractMaxMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        prod = membership_i * membership_j
        return prod


class FuzzyMaxMinMerger(FuzzyAbstractMaxMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        minm = np.minimum(membership_i, membership_j)
        return minm


class FuzzyMaxGMMerger(FuzzyAbstractMaxMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        GM = np.sqrt(membership_i * membership_j)
        return GM


class FuzzyMaxOBMerger(FuzzyAbstractMaxMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        OB = np.sqrt((membership_i * membership_j) * np.minimum(membership_i, membership_j))
        return OB

class FuzzyMaxODivMerger(FuzzyAbstractMaxMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        ODiv = (membership_i * membership_j + np.minimum(membership_i, membership_j)) / 2
        return ODiv


# Dual GM
class FuzzyAbstractDGMMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics,dims=4)

    def _merge_matrix(self, i, j):
        # Initialization
        self.similMatrix[:i, i, 3] = np.maximum(
            self.similMatrix[:i, i, 3],
            self.similMatrix[:i, j, 3])
        self.similMatrix[i, i + 1:j, 3] = np.maximum(
            self.similMatrix[i, i + 1:j, 3],
            self.similMatrix[i + 1:j, j, 3])
        self.similMatrix[i, j + 1:, 3] = np.maximum(
            self.similMatrix[i, j + 1:, 3],
            self.similMatrix[j, j + 1:, 3])
        # Cardinality
        self.similMatrix[:i, i, 2] = (
            self.similMatrix[:i, i, 2] +
            self.similMatrix[:i, j, 2])
        self.similMatrix[i, i + 1:j, 2] = (
            self.similMatrix[i, i + 1:j, 2] +
            self.similMatrix[i + 1:j, j, 2])
        self.similMatrix[i, j + 1:, 2] = (
            self.similMatrix[i, j + 1:, 0] +
            self.similMatrix[j, j + 1:, 0])
        # Product
        self.similMatrix[:i, i, 1] = (
            self.similMatrix[:i, i, 1] *
            self.similMatrix[:i, j, 1])
        self.similMatrix[i, i + 1:j, 1] = (
            self.similMatrix[i, i + 1:j, 1] *
            self.similMatrix[i + 1:j, j, 1])
        self.similMatrix[i, j + 1:, 1] = (
            self.similMatrix[i, j + 1:, 1] *
            self.similMatrix[j, j + 1:, 1])
        # Similarity
        self.similMatrix[:i, i, 0] = (1 -
            self.similMatrix[:i, i, 1] **
            (1 / self.similMatrix[:i, j, 2]))
        self.similMatrix[i, i + 1:j, 0] = (1 -
            self.similMatrix[i, i + 1:j, 1] **
            (1 / self.similMatrix[i + 1:j, j, 2]))
        self.similMatrix[i, j + 1:, 0] = (1 -
            self.similMatrix[i, j + 1:, 1] **
            (1 / self.similMatrix[j, j + 1:, 2]))

    def _func(self, membership_i, membership_j):
        pass

    def _similarity(self, fmics, memberships, i, j):
        result = self._func(memberships[i], memberships[j])
        if self.similMatrix[i, j, 3] == 0:
            self.similMatrix[i, j, 1] = (1 - result)
            self.similMatrix[i, j, 2] = 1
            self.similMatrix[i, j, 3] = 1
        else:
            self.similMatrix[i, j, 1] *=  (1 - result)
            self.similMatrix[i, j, 2] += 1
        self.similMatrix[i, j, 0] = 1 - self.similMatrix[i, j, 1]**(1 / self.similMatrix[i, j, 2])
        similarity = self.similMatrix[i, j, 0]
        return similarity


class FuzzyDGMProdMerger(FuzzyAbstractDGMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        prod = membership_i * membership_j
        return prod


class FuzzyDGMMinMerger(FuzzyAbstractDGMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        minm = np.minimum(membership_i, membership_j)
        return minm


class FuzzyDGMGMMerger(FuzzyAbstractDGMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        GM = np.sqrt(membership_i * membership_j)
        return GM


class FuzzyDGMOBMerger(FuzzyAbstractDGMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        OB = np.sqrt((membership_i * membership_j) * np.minimum(membership_i, membership_j))
        return OB

class FuzzyDGMODivMerger(FuzzyAbstractDGMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        ODiv = (membership_i * membership_j + np.minimum(membership_i, membership_j))/2
        return ODiv


# GB
class FuzzyAbstractGBMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics, dims=4)

    def _merge_matrix(self, i, j):
        # Initialization
        self.similMatrix[:i, i, 3] = np.maximum(
            self.similMatrix[:i, i, 3],
            self.similMatrix[:i, j, 3])
        self.similMatrix[i, i + 1:j, 3] = np.maximum(
            self.similMatrix[i, i + 1:j, 3],
            self.similMatrix[i + 1:j, j, 3])
        self.similMatrix[i, j + 1:, 3] = np.maximum(
            self.similMatrix[i, j + 1:, 3],
            self.similMatrix[j, j + 1:, 3])
        # Cardinality
        self.similMatrix[:i, i, 2] = (
            self.similMatrix[:i, i, 2] +
            self.similMatrix[:i, j, 2])
        self.similMatrix[i, i + 1:j, 2] = (
            self.similMatrix[i, i + 1:j, 2] +
            self.similMatrix[i + 1:j, j, 2])
        self.similMatrix[i, j + 1:, 2] = (
            self.similMatrix[i, j + 1:, 0] +
            self.similMatrix[j, j + 1:, 0])
        # Product
        self.similMatrix[:i, i, 1] = (
            self.similMatrix[:i, i, 1] *
            self.similMatrix[:i, j, 1])
        self.similMatrix[i, i + 1:j, 1] = (
            self.similMatrix[i, i + 1:j, 1] *
            self.similMatrix[i + 1:j, j, 1])
        self.similMatrix[i, j + 1:, 1] = (
            self.similMatrix[i, j + 1:, 1] *
            self.similMatrix[j, j + 1:, 1])
        # Similarity
        self.similMatrix[:i, i, 0] = (1 -
            (self.similMatrix[:i, i, 1] *
            self.similMatrix[:i, j, 2])**(1/2))
        self.similMatrix[i, i + 1:j, 0] = (1 -
            (self.similMatrix[i, i + 1:j, 1] *
             self.similMatrix[i + 1:j, j, 2])**(1/2))
        self.similMatrix[i, j + 1:, 0] = (1 -
            (self.similMatrix[i, j + 1:, 1] *
             self.similMatrix[j, j + 1:, 2])**(1/2))

    def _func(self, membership_i, membership_j):
        pass

    def _similarity(self, fmics, memberships, i, j):
        result = self._func(memberships[i], memberships[j])
        if self.similMatrix[i, j, 3] == 0:
            self.similMatrix[i, j, 1] = (1 - result)
            self.similMatrix[i, j, 2] = (1 - result)
            self.similMatrix[i, j, 3] = 1
        else:
            self.similMatrix[i, j, 1] *= (1 - result)
            self.similMatrix[i, j, 2] = np.minimum(self.similMatrix[i, j, 2], 1 - result)
        self.similMatrix[i, j, 0] = 1 - (self.similMatrix[i, j, 1] * self.similMatrix[i, j, 2])**(1/2)
        similarity = self.similMatrix[i, j, 0]
        return similarity


class FuzzyGBProdMerger(FuzzyAbstractGBMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        prod = membership_i * membership_j
        return prod


class FuzzyGBMinMerger(FuzzyAbstractGBMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        minm = np.minimum(membership_i, membership_j)
        return minm


class FuzzyGBGMMerger(FuzzyAbstractGBMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        GM = np.sqrt(membership_i * membership_j)
        return GM


class FuzzyGBOBMerger(FuzzyAbstractGBMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        OB = np.sqrt((membership_i * membership_j) * np.minimum(membership_i, membership_j))
        return OB

class FuzzyGBODivMerger(FuzzyAbstractGBMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        ODiv = (membership_i * membership_j + np.minimum(membership_i, membership_j))/2
        return ODiv


# Dual GDIV
class FuzzyAbstractGDIVMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics, dims=4)

    def _merge_matrix(self, i, j):
        # Initialization
        self.similMatrix[:i, i, 3] = np.maximum(
            self.similMatrix[:i, i, 3],
            self.similMatrix[:i, j, 3])
        self.similMatrix[i, i + 1:j, 3] = np.maximum(
            self.similMatrix[i, i + 1:j, 3],
            self.similMatrix[i + 1:j, j, 3])
        self.similMatrix[i, j + 1:, 3] = np.maximum(
            self.similMatrix[i, j + 1:, 3],
            self.similMatrix[j, j + 1:, 3])
        # Minimum
        self.similMatrix[:i, i, 2] = np.minimum(
            self.similMatrix[:i, i, 2],
            self.similMatrix[:i, j, 2])
        self.similMatrix[i, i + 1:j, 2] = np.minimum(
            self.similMatrix[i, i + 1:j, 2],
            self.similMatrix[i + 1:j, j, 2])
        self.similMatrix[i, j + 1:, 2] = np.minimum(
            self.similMatrix[i, j + 1:, 2],
            self.similMatrix[j, j + 1:, 2])
        # Product
        self.similMatrix[:i, i, 1] = (
            self.similMatrix[:i, i, 1] *
            self.similMatrix[:i, j, 1])
        self.similMatrix[i, i + 1:j, 1] = (
            self.similMatrix[i, i + 1:j, 1] *
            self.similMatrix[i + 1:j, j, 1])
        self.similMatrix[i, j + 1:, 1] = (
            self.similMatrix[i, j + 1:, 1] *
            self.similMatrix[j, j + 1:, 1])
        # Similarity
        self.similMatrix[:i, i, 0] = (1 -
            (self.similMatrix[:i, i, 1] +
             self.similMatrix[:i, j, 2]) / 2)
        self.similMatrix[i, i + 1:j, 0] = (1 -
            (self.similMatrix[i, i + 1:j, 1] +
             self.similMatrix[i + 1:j, j, 2]) / 2)
        self.similMatrix[i, j + 1:, 0] = (1 -
            (self.similMatrix[i, j + 1:, 1] +
             self.similMatrix[j, j + 1:, 2]) / 2)

    def _func(self, membership_i, membership_j):
        pass

    def _similarity(self, fmics, memberships, i, j):
        result = self._func(memberships[i], memberships[j])
        if self.similMatrix[i, j, 3] == 0:
            self.similMatrix[i, j, 1] = (1 - result)
            self.similMatrix[i, j, 2] = (1 - result)
            self.similMatrix[i, j, 3] = 1
        else:
            self.similMatrix[i, j, 1] *= (1 - result)
            self.similMatrix[i, j, 2] = np.minimum(self.similMatrix[i, j, 2], 1 - result)
        self.similMatrix[i, j, 0] = 1 - (self.similMatrix[i, j, 1] * self.similMatrix[i, j, 2]) / 2
        similarity = self.similMatrix[i, j, 0]
        return similarity


class FuzzyGDIVProdMerger(FuzzyAbstractGDIVMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        prod = membership_i * membership_j
        return prod


class FuzzyGDIVMinMerger(FuzzyAbstractGDIVMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        minm = np.minimum(membership_i, membership_j)
        return minm


class FuzzyGDIVGMMerger(FuzzyAbstractGDIVMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        GM = np.sqrt(membership_i * membership_j)
        return GM


class FuzzyGDIVOBMerger(FuzzyAbstractGDIVMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        OB = np.sqrt((membership_i * membership_j) * np.minimum(membership_i, membership_j))
        return OB

class FuzzyGDIVODivMerger(FuzzyAbstractGDIVMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        ODiv = (membership_i * membership_j + np.minimum(membership_i, membership_j))/2
        return ODiv


# Arithmetic Mean
class FuzzyAbstractAMMerger(AbstractFuzzyDissimilarityMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _merge_matrix(self, i, j):
        # Cardinality
        self.similMatrix[:i, i, 2] = (
            self.similMatrix[:i, i, 2] +
            self.similMatrix[:i, j, 2])
        self.similMatrix[i, i + 1:j, 2] = (
            self.similMatrix[i, i + 1:j, 2] +
            self.similMatrix[i + 1:j, j, 2])
        self.similMatrix[i, j + 1:, 2] = (
            self.similMatrix[i, j + 1:, 2] +
            self.similMatrix[j, j + 1:, 2])
        # Sum
        self.similMatrix[:i, i, 1] = (
            self.similMatrix[:i, i, 1] +
            self.similMatrix[:i, j, 1])
        self.similMatrix[i, i + 1:j, 1] = (
            self.similMatrix[i, i + 1:j, 1] +
            self.similMatrix[i + 1:j, j, 1])
        self.similMatrix[i, j + 1:, 1] = (
            self.similMatrix[i, j + 1:, 1] +
            self.similMatrix[j, j + 1:, 1])
        # Similarity
        self.similMatrix[:i, i, 0] = (
            (self.similMatrix[:i, i, 1] /
             self.similMatrix[:i, j, 2]))
        self.similMatrix[i, i + 1:j, 0] = (
            (self.similMatrix[i, i + 1:j, 1] /
             self.similMatrix[i + 1:j, j, 2]))
        self.similMatrix[i, j + 1:, 0] = (
            (self.similMatrix[i, j + 1:, 1] /
             self.similMatrix[j, j + 1:, 2]))

    def _func(self, membership_i, membership_j):
        pass

    def _similarity(self, fmics, memberships, i, j):
        result = self._func(memberships[i], memberships[j])
        self.similMatrix[i, j, 1] += result
        self.similMatrix[i, j, 2] += 1
        self.similMatrix[i, j, 0] = self.similMatrix[i, j, 1] / self.similMatrix[i, j, 2]
        similarity = self.similMatrix[i, j, 0]
        return similarity


class FuzzyAMProdMerger(FuzzyAbstractAMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        prod = membership_i * membership_j
        return prod


class FuzzyAMMinMerger(FuzzyAbstractAMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        minm = np.minimum(membership_i, membership_j)
        return minm


class FuzzyAMGMMerger(FuzzyAbstractAMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        GM = np.sqrt(membership_i * membership_j)
        return GM


class FuzzyAMOBMerger(FuzzyAbstractAMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        OB = np.sqrt((membership_i * membership_j) * np.minimum(membership_i, membership_j))
        return OB

class FuzzyAMODivMerger(FuzzyAbstractAMMerger):

    def __init__(self, sm, threshold, max_fmics):
        super().__init__(sm, threshold, max_fmics)

    def _func(self, membership_i, membership_j):
        ODiv = (membership_i * membership_j + np.minimum(membership_i, membership_j))/2
        return ODiv


AllMergers = {
    1: FuzzyEuclideanMerger,
    2: FuzzyMinMaxMerger,
    3: FuzzyREFMerger,
    4: FuzzyPSProdMerger,
    5: FuzzyPSMinMerger,
    6: FuzzyPSGMMerger,
    7: FuzzyPSOBMerger,
    8: FuzzyPSODivMerger,
    9: FuzzyMaxProdMerger,
    10: FuzzyMaxMinMerger,
    11: FuzzyMaxGMMerger,
    12: FuzzyMaxOBMerger,
    13: FuzzyMaxODivMerger,
    14: FuzzyDGMProdMerger,
    15: FuzzyDGMMinMerger,
    16: FuzzyDGMGMMerger,
    17: FuzzyDGMOBMerger,
    18: FuzzyDGMODivMerger,
    19: FuzzyGBProdMerger,
    20: FuzzyGBMinMerger,
    21: FuzzyGBGMMerger,
    22: FuzzyGBOBMerger,
    23: FuzzyGBODivMerger,
    24: FuzzyGDIVProdMerger,
    25: FuzzyGDIVMinMerger,
    26: FuzzyGDIVGMMerger,
    27: FuzzyGDIVOBMerger,
    28: FuzzyGDIVODivMerger,
    29: FuzzyAMProdMerger,
    30: FuzzyAMMinMerger,
    31: FuzzyAMGMMerger,
    32: FuzzyAMOBMerger,
    33: FuzzyAMODivMerger,
    }

# OLD MONOLITHIC VERSION
class FuzzyDissimilarityMerger:
    def __init__(self, sm, threshold, max_fmics):
        self.similMatrix = np.zeros((max_fmics, max_fmics, 2))  # FIXME: Changed from 3
        self.auxMatrix = np.zeros((max_fmics, max_fmics, 2))
        self.threshold = threshold  # NOTE: Before it was an argument for
        # merge. It would allow to change mid run, but we don't do it

        # Self.similMatrix.flat[0::6] = 1
        self.sm = sm

    def merge(self, fmics, memberships):
        fmics_to_merge = []

        for i in range(0, len(fmics) - 1):
            for j in range(i + 1, len(fmics)):
                # Similarity S1 - euclidean
                if (self.sm == 1):
                    dissimilarity = EuclideanDistance.distance(fmics[i].center, fmics[j].center)
                    sum_of_radius = fmics[i].radius + fmics[j].radius
                    if dissimilarity != 0:
                        similarity = sum_of_radius / dissimilarity
                    else:
                        # Highest value possible
                        similarity = 1.7976931348623157e+308

                # Similarity S2 - SUMmin/SUMmax
                elif (self.sm == 2):
                    self.similMatrix[i, j, 0] += np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 1] += np.maximum(memberships[i], memberships[j])
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]

                #================================================================================================================================
                            # S(A,B) = AM(REF(x_1,y_1), ... REF(x_n, y_n))
                #================================================================================================================================
                elif (self.sm == 3):
                    t = 2
                    self.similMatrix[i, j, 0] += np.power(1 - np.absolute(memberships[i] - memberships[j]), 1/t)
                    self.similMatrix[i, j, 1] += 1 # NAO È A MÉDIA É O NUMERO DE PONTOS!!!
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]

                #================================================================================================================================
                            # S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = Probabilistic Sum (idx 4 to 8)
                #================================================================================================================================
                # O = Product
                elif (self.sm == 4):
                    Prod = memberships[i] * memberships[j]
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + Prod - self.similMatrix[i, j, 0] * Prod
                    similarity = self.similMatrix[i, j, 0]

                # O = MIN
                elif (self.sm == 5):
                    min = np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + min - self.similMatrix[i, j, 0] * min
                    similarity = self.similMatrix[i, j, 0]

                # O = GM
                elif (self.sm == 6):
                    GM = np.sqrt(memberships[i] * memberships[j])
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + GM - self.similMatrix[i, j, 0] * GM
                    similarity = self.similMatrix[i, j, 0]

                # O = OB
                elif (self.sm == 7):
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum(memberships[i], memberships[j]))
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + OB - self.similMatrix[i, j, 0] * OB
                    similarity = self.similMatrix[i, j, 0]

                # O = ODiv
                elif (self.sm == 8):
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2
                    self.similMatrix[i, j, 0] = self.similMatrix[i, j, 0] + ODiv - self.similMatrix[i, j, 0] * ODiv
                    similarity = self.similMatrix[i, j, 0]

                #================================================================================================================================
                            # S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = Maximum (idx 9 to 13)
                #================================================================================================================================
                # O = Product
                elif (self.sm == 9):
                    Prod = memberships[i] * memberships[j]
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], Prod)
                    similarity = self.similMatrix[i, j, 0]

                # O = MIN
                elif (self.sm == 10):
                    min = np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], min)
                    similarity = self.similMatrix[i, j, 0]

                # O = GM
                elif (self.sm == 11):
                    GM = np.sqrt(memberships[i] * memberships[j])
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], GM)
                    similarity = self.similMatrix[i, j, 0]

                # O = OB
                elif (self.sm == 12):
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum(memberships[i], memberships[j]))
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], OB)
                    similarity = self.similMatrix[i, j, 0]

                # O = ODiv
                elif (self.sm == 13):
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2
                    self.similMatrix[i, j, 0] = np.maximum(self.similMatrix[i, j, 0], ODiv)
                    similarity = self.similMatrix[i, j, 0]

                #================================================================================================================================
                            # S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = G(GM) (idx 14 to 18)
                            # G(GM) = 1 - ((1-GM(x,y))^n * (1 - atual))^1/n+1
                #================================================================================================================================
                # O = Product
                elif (self.sm == 14):
                    Prod = memberships[i] * memberships[j]
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - Prod
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1]

                    else:
                        # Similarity
                        self.auxMatrix[i, j, 1] *= 1 - Prod
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1], 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                # O = MIN
                elif (self.sm == 15):
                    min = np.minimum(memberships[i], memberships[j])
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - min
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1]

                    else:
                        # Similarity
                        self.auxMatrix[i, j, 1] *= 1 - min
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1], 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                # O = GM
                elif (self.sm == 16):
                    GM = np.sqrt(memberships[i] * memberships[j])
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - GM
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1]

                    else:
                        # Similarity
                        self.auxMatrix[i, j, 1] *= 1 - GM
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1], 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                # O = OB
                elif (self.sm == 17):
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum(memberships[i], memberships[j]))
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - OB
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1]

                    else:
                        # Similarity
                        self.auxMatrix[i, j, 1] *= 1 - OB
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1], 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                # O = ODiv
                elif (self.sm == 18):
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2
                    if (self.similMatrix[i, j, 1] == 0):
                        # n = 0
                        self.auxMatrix[i, j, 1] = 1 - ODiv
                        self.similMatrix[i, j, 0] = 1 - self.auxMatrix[i, j, 1]

                    else:
                        # Similarity
                        self.auxMatrix[i, j, 1] *= 1 - ODiv
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1], 1/self.similMatrix[i, j, 1]+1)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                #================================================================================================================================
                            # S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = Dual(OB) (idx 19 to 23)
                #================================================================================================================================
                # O = Product
                elif (self.sm == 19):
                    Prod = memberships[i] * memberships[j]
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - min
                        self.auxMatrix[i, j, 1] = 1 - Prod
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)

                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.auxMatrix[i, j, 1] *= 1 - Prod
                        # GB =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0]

                # O = MIN
                elif (self.sm == 20):
                    min = np.minimum(memberships[i], memberships[j])
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - min
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 0] * self.auxMatrix[i, j, 0], 1/2)
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 0] * self.auxMatrix[i, j, 0], 1/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]

                # O = GM
                elif (self.sm == 21):
                    GM = np.sqrt(memberships[i] * memberships[j])
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - min
                        self.auxMatrix[i, j, 1] = 1 - GM
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)

                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.auxMatrix[i, j, 1] *= 1 - GM
                        # GB =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]

                # O = OB
                elif (self.sm == 22):
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum(memberships[i], memberships[j]))
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - min
                        self.auxMatrix[i, j, 1] = 1 - OB
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)

                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.auxMatrix[i, j, 1] *= 1 - OB
                        # GB =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]

                # O = ODiv
                elif (self.sm == 23):
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - min
                        self.auxMatrix[i, j, 1] = 1 - ODiv
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)

                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.auxMatrix[i, j, 1] *= 1 - ODiv
                        # GB =  1 - (PROD * MIN)^{1/2}
                        self.similMatrix[i, j, 0] = 1 - np.power(self.auxMatrix[i, j, 1] * self.auxMatrix[i, j, 0], 1/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]


                #================================================================================================================================
                            # S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = Dual(ODiv) (idx 24 to 28)
                #================================================================================================================================
                # O = Product
                elif (self.sm == 24):
                    Prod = memberships[i] * memberships[j]
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - Prod
                        self.auxMatrix[i, j, 1] = 1 - Prod
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - Prod)
                        self.auxMatrix[i, j, 1] *= 1 - Prod
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]

                # O = MIN
                elif (self.sm == 25):
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - min
                        self.auxMatrix[i, j, 1] = 1 - min
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.auxMatrix[i, j, 1] *= 1 - min
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]

                # O = GM
                elif (self.sm == 26):
                    GM = np.sqrt(memberships[i] * memberships[j])
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - GM
                        self.auxMatrix[i, j, 1] = 1 - GM
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.auxMatrix[i, j, 1] *= 1 - GM
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]

                # O = OB
                elif (self.sm == 27):
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum(memberships[i], memberships[j]))
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - OB
                        self.auxMatrix[i, j, 1] = 1 - OB
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.auxMatrix[i, j, 1] *= 1 - OB
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]

                # O = ODiv
                elif (self.sm == 28):
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum(memberships[i], memberships[j]))
                    min = np.minimum(memberships[i], memberships[j])
                    # Se o n=0 é o primeiro caso
                    if (self.similMatrix[i, j, 1] == 0):
                        self.auxMatrix[i, j, 0] = 1 - ODiv
                        self.auxMatrix[i, j, 1] = 1 - ODiv
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    # Senão n>=1
                    else:
                        self.auxMatrix[i, j, 0] = 1 - np.minimum(self.auxMatrix[i, j, 0], 1 - min)
                        self.auxMatrix[i, j, 1] *= 1 - ODiv
                        self.similMatrix[i, j, 0] = 1 - ((self.auxMatrix[i, j, 1] + self.auxMatrix[i, j, 0])/2)
                    self.similMatrix[i, j, 1] +=1
                    similarity = self.similMatrix[i, j, 0]

                #================================================================================================================================
                            # S(A,B) = G(O(x_1,y_1), ... O(x_n, y_n))    -> G = MÉDIA (idx 29 to 33)
                #================================================================================================================================
                # O = Product
                elif (self.sm == 29):
                    Prod = memberships[i] * memberships[j]
                    self.similMatrix[i, j, 0] += Prod
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]

                # O = MIN
                elif (self.sm == 30):
                    min = np.minimum(memberships[i], memberships[j])
                    self.similMatrix[i, j, 0] += min
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]

                # O = GM
                elif (self.sm == 31):
                    GM = np.sqrt(memberships[i] * memberships[j])
                    self.similMatrix[i, j, 0] += GM
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]

                # O = OB
                elif (self.sm == 32):
                    OB = np.sqrt((memberships[i] * memberships[j]) * np.minimum(memberships[i], memberships[j]))
                    self.similMatrix[i, j, 0] += OB
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]

                # O = ODiv
                elif (self.sm == 33):
                    ODiv = (memberships[i] * memberships[j] + np.minimum(memberships[i], memberships[j]))/2
                    self.similMatrix[i, j, 0] += ODiv
                    self.similMatrix[i, j, 1] += 1
                    similarity = self.similMatrix[i, j, 0] / self.similMatrix[i, j, 1]

                if similarity >= self.threshold:
                    fmics_to_merge.append([i, j, similarity])

        # Sort by most similar
        fmics_to_merge.sort(reverse=True, key=lambda k: k[2])
        merged_fmics_idx = []
        merged_fmics = []

        for (i, j, _) in fmics_to_merge:
            if i not in merged_fmics_idx and j not in merged_fmics_idx:
                merged_fmics.append(FMiC.merge(fmics[i], fmics[j]))
                merged_fmics_idx.append(i)
                merged_fmics_idx.append(j)

        merged_fmics_idx.sort(reverse=True)
        for idx in merged_fmics_idx:
            fmics.pop(idx)

        return fmics + merged_fmics
