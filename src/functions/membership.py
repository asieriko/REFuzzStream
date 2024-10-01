class FuzzyCMeansMembership:

    def memberships(distances, m):
        memberships = []
        for distance_j in distances:
            # To avoid division by 0
            sum_of_distances = 0
            for distance_k in distances:
                if distance_k != 0:
                    sum_of_distances += pow((distance_j / distance_k), 2. / (m - 1.))
            if sum_of_distances != 0:  # If the distance is 0, membership should be 1 (It happens in the powersupply ds)
                memberships.append(1.0 / sum_of_distances)
            else:
                memberships.append(1.0)
        return memberships
