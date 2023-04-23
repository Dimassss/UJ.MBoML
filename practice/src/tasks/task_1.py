import numpy as np
import pandas as pd
import math


def nearest(x, k, data):
    # return a predicted class for the new data piont x based on k-nearest neighbor algorithm for dataset data

    if len(data) <= k:
        return data;

    closest_points = data[0:k]

    def dist(a,b):
        return np.linalg.norm(a-b)

    data_dists = [dist(a,x) for a in data]
    best_k_dists = list(range(k))

    # Possible To Remove Point
    def get_ptr_point():
        d = [data_dists[i] for i in best_k_dists[0:k]]
        return d.index(max(d))

    ptr_point = get_ptr_point()

    for i in range(k, len(data)):
        if data_dists[i] < data_dists[best_k_dists[ptr_point]]:
            best_k_dists[ptr_point] = i
            ptr_point = get_ptr_point()

    return np.array([data[i] for i in best_k_dists])
