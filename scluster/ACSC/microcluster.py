import copy
import numpy as np

class microcluster():

    def __init__(self):
        self.N = 0
        self.LS = 0
        self.SS = 0
        self.r = 0
        self.c = []

    def update(self):
        self.c = self.LS / self.N
        self.r = np.sqrt(self.SS/self.N - self.c**2)  # it is an array, not an scalar
    
    def add_point(self, x: np.ndarray):
        self.N += 1
        self.LS += x
        self.SS += x**2
        self.update()

    def add_microcluster(self, b):
        self.N += b.N
        self.LS += b.LS
        self.SS += b.SS
        self.update()

    def __str__(self):
        return f"{self.c}, {self.N} points"

    def __repr__(self):
        return f"{self.c}, {self.r}, {self.N}"


def merge_microclusters(a, b, epsilon=0.05):
    c = microcluster()
    c = copy.deepcopy(a)
    c.add_microcluster(b)
    r = np.sqrt(sum(c.r**2)) 
    if r <= epsilon:
        # delete a: outside
        # delete b: outside
        return c
    else:
        # delete c
        return False
