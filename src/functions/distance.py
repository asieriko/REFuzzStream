from math import sqrt


class EuclideanDistance:
    
    def distance(value_a, value_b):
        sum_of_distances = 0
        for idx, value in enumerate(value_a):
            sum_of_distances += pow(value - value_b[idx], 2)
        return sqrt(sum_of_distances)
