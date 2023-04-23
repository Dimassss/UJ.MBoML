import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from src.tasks.task_1 import nearest

def KNN_classify(knn_best_points_labels):
    # knn_best_points_labels is 1-dim array of labels of primitive type. 
    # it will return label which has the biggest number of occurances in array
    
    labels = []
    occurances = []

    for k in knn_best_points_labels:
        try:
            i = labels.index(k)
            occurances[i] += 1
        except ValueError:
            labels += [k]
            occurances += [0]

    return labels[occurances.index(max(occurances))]



def draw_plot(k, data, n):
    # k is k in KNN
    # data is points which will be ploted
    # n is number of squares/pizels per side
    # this function will draw plot of size n*n squares where each square will be colred in different color
    # depending on class to which that point/square will be classiefied by KNN method.

    OX = np.linspace(min(data[:, 0]), max(data[:, 0]), n)
    OY = np.linspace(min(data[:, 1]), max(data[:, 1]), n)

    X, Y = np.meshgrid(OX, OY)
    
    def get_label(i, j):
        nonlocal k
        
        best_points = nearest([X[i, j], Y[i, j]], k, data[:, 0:2])
        best_points_map = []
    
        for f in range(len(best_points)):
            bp = best_points[f]
            for p in data:
                if p[0] == bp[0] and p[1] == bp[1]:
                    best_points_map += [p]
                    break

        label = KNN_classify(np.array(best_points_map)[:, 2])
        print(label)
        return label

    Z = [[get_label(i, j) for i in range(n)] for j in range(n)]

    plt.contourf(X, Y, Z)
    plt.colorbar()

