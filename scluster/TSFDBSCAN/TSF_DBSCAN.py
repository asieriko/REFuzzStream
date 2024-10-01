#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:17:25 2023

@author: asier
"""
import math


class p_object():
    '''
    x: the values of the object

    t: the timestamp of arrival. For the purposes of this article, it can be
       considered as an incremental index;

    kernel: the set of objects that belong to the kernel neighborhood of p,
            i.e., that lie at a distance lower than, or equal to, εmin from p;

    shell: the set of objects that belong to the shell-neighborhood of p,
           i.e., that lie at a distance between εmin and εmax (included) from p;

    weight: the weight of the object, which is set to 1 at the arrival time and
            decreases over time according to the window model. Details about
            the adopted damped window model are given in Section IV-C1;

    memberships: a data structure that associates the membership degree of a
                 border object p with each of the close clusters.
                As an example, given a set of three clusters C1, C2, C3, then
                  p.memberships={C1:0.6,C3:0.1}
                  View SourceRight-click on figure for MathML and additional
                  features.means that p belongs to C1 with a membership degree of
                  0.6 and to C3 with a membership degree of 0.1. It follows that:
                  p.memberships[C1]=0.6
                  p.memberships[C2]=0
                  p.memberships[C3]=0.1.
    '''

    def __init__(self, x, t=0, weight=1):
        self.x = x
        self.t = t
        self.weight = weight
        self.kernel = []
        self.shell = []
        self.memberships = {}
        self.outlier = True  # FIXME: it is not used ~ memberships = {}
        self.in_kernels = []
        self.in_shells = []

    def remove_from_kernel(self, p):
        self.kernel.remove(p)

    def remove_from_shell(self, p):
        self.shell.remove(p)

    def get_membership(self, cluster):
        if cluster.id not in self.memberships.keys():
            return 0
        else:
            return self.memberships[cluster.id]

    def get_max_cluster_membership(self):
        if self.memberships != {}:
            cluster = max(self.memberships, key=self.memberships.get)
            return cluster, self.memberships[cluster]
        return -1, 1

    def __repr__(self) -> str:
        return f"p_object({self.x}, {self.t}, {self.weight})"

    def __str__(self) -> str:
        return f"Point: {self.x} at time {self.t}"


class Cluster():

    c_id = 0

    def __init__(self):
        self.core_points = []
        self.id = Cluster.c_id
        Cluster.c_id += 1

    def add_core_point(self, point):
        point.memberships[self.id] = 1
        self.core_points.append(point)

    def point_in_cluster(self, point) -> bool:
        return point in self.core_points

    def __repr__(self) -> str:
        return f"Cluster ({self.core_points})"

    def __str__(self) -> str:
        return f"Cluster: {len(self.core_points)} elements"


class TSF_DBSCAN():

    def __init__(self, emin, emax, alpha, Ow, MinWeight, toffline):
        self.plist = []
        self.emin = emin
        self.emax = emax
        self.alpha = alpha
        self.Ow = Ow
        self.MinWeight = MinWeight
        self.toffline = toffline
        self.distance = lambda x, y: math.sqrt(sum(map(lambda a, b: (a - b)**2, x.x, y.x)))
        self.clusters = []  # I wonder ?

    def fuzzyQuery(self, p):
        '''
        Given the parameters εmin and εmax, the procedure computes the set of objects
        to be inserted into p.kernel and p.shell.
        The procedure extends the RegionQuery procedure defined in [7] by relaxing
        the crisp constraint on the radius ε.

        εmin: the kernel-neighborhood radius of an object;

        εmax: maximum radius that, along with εmin, defines the shell neighborhood
              of an object;
        '''
        kernel = []
        shell = []

        for q in self.plist:
            if self.distance(p, q) < self.emin:
                kernel.append(q)
            elif self.distance(p, q) < self.emax:
                shell.append(q)

        return kernel, shell

    def evaluateCore(self, p, t):
        '''
        p - candidate core object

        t - actual timestamp, depending on the last arrived object

        alpha: decay factor adopted for the damped window model;

        Ow: weight threshold. Objects with a weight lower than Ow are removed.
        '''
        weightSum = 2**(-1 * self.alpha * (t - p.t))
        Oth = t + math.log(self.Ow, 2) / self.alpha
        for s in p.shell[:]:
            if s.t < Oth:
                p.remove_from_shell(s)
        for k in p.kernel[:]:
            if k.t < Oth:
                p.remove_from_kernel(k)
            else:
                k.weight = 2**(-1 * self.alpha * (t - k.t))
                weightSum = weightSum + k.weight
        if weightSum >= self.MinWeight:
            return True
        else:
            return False

    def fuzzyMembership(self, b, c):
        '''
        membership
        1 | -----\
          |     . \
        0 |     .  \-------
          --------------------
                |  |     distance
            emin  |
                  emax
        '''
        dist = self.distance(b, c)
        if dist <= self.emin:
            return 1
        elif dist >= self.emax:
            return 0
        else:
            return 1 - ((self.emin - dist) / (self.emin - self.emax))

    def fuzzyBorderUpdate(self, b, c, C):
        '''
        b - border object

        c - core object
        '''
        mbc = self.fuzzyMembership(b, c)
        b.memberships[C.id] = max(b.get_membership(C), mbc)  # id to identify each cluster
        # for cluster in self.clusters:
            # if cluster.point_in_cluster(c):
            #     C = cluster
            #     C, _ = c.get_max_cluster_membership()
                # b.memberships[C.id] = max(b.get_membership(C), mbc)  # id to identify each cluster
                # if b.membership[C.id]: #  ¿? b.outlier = False
                # break

    def offline_fdbscan(self, t):
        '''
        pList - list of valid arrived objects, not yet expired

        t - current timestamp, depending on the last arrived object

        alpha: decay factor adopted for the damped window model;

        Ow: weight threshold. Objects with a weight lower than Ow are removed.
        '''
        self.clusters = []  # I wonder ?
        Cluster._Cluster_id = 0  # Reset the cluster id names
        Oth = t + math.log(self.Ow, 2) / self.alpha
        for q in self.plist[:]:
            if q.t < Oth:
                self.plist.remove(q)
            else:
                q.outlier = True
                q.memberships = {}
        visited = []
        for p in self.plist:
            to_visit = []
            first = True
            if p not in visited:
                to_visit = to_visit + [p]
                # for q in to_visit:
                while len(to_visit) > 0:  # Asier
                    q = to_visit.pop(0)
                    if self.evaluateCore(q, t):
                        if q not in visited:
                            visited = visited + [q]
                            if first is True:
                                C = Cluster()
                                self.clusters.append(C)  # I wonder ?
                                first = False
                            C.add_core_point(q)  # ¿? q.outlier = False
                            to_visit = to_visit + q.kernel
                            for s in q.kernel + q.shell:
                                '''
                                An object is a core object if the sum of the weights of the
                                objects in its kernel-neighborhood is higher than a prefixed
                                threshold. An object that does not satisfy the core condition
                                but lies in the neighborhood of one or more core objects is
                                assigned to its or their fuzzy neighborhood, respectively.
                                '''
                                # if not isCore(s): #
                                # if s not in C.core_points:
                                if not self.evaluateCore(s, t):
                                    self.fuzzyBorderUpdate(s, q, C)

    def tsfdbscan(self, p):
        '''
        p: new object

        plist: list of valid arrived objects, not yet expired

        εmin: the kernel-neighborhood radius of an object;

        εmax: maximum radius that, along with εmin, defines the shell neighborhood
              of an object;

        Toffline: period of the offline cluster evaluation;
        '''
        # At the arrival of each object p of the stream,
        # first of all, the fuzzyQuery procedure is called.
        p.kernel, p.shell = self.fuzzyQuery(p)
        # Then, for each object k in p.kernel, and for each object s in p.shell,
        # k.kernel and s.shell are updated by adding object p.

        for k in p.kernel:
            k.kernel = k.kernel + [p]
        for s in p.shell:
            s.shell = s.shell + [p]
        self.plist = self.plist + [p]
        # Finally, if the timestamp p.t is a multiple of the period Toffline,
        # then the offline reclustering stage is triggered.
        if (p.t + 1) % self.toffline == 0:
            self.offline_fdbscan(p.t)
