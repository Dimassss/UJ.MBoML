import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

matplotlib.use('TkAgg')

def task_1():
    iris = datasets.load_iris()
    X = iris.data[:,:2]
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3)
    model.fit(X_scaled)

    centers = model.cluster_centers_

    plt.figure(figsize=(12, 6))

    N = 1000
    ox = np.linspace(np.min(X_scaled[:, 0]), np.max(X_scaled[:, 0]), N)
    oy = np.linspace(np.min(X_scaled[:, 1]), np.max(X_scaled[:, 1]), N)
    OX, OY = np.meshgrid(ox, oy)
    Z = [model.predict([[OX[i,j], OY[i,j]] for j in range(N)]) for i in range(N)]

    plt.contourf(OX, OY, Z)

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.6)
    plt.xlabel('Scaled feature 1')
    plt.ylabel('Scaled feature 2')

    plt.show()

def main():
    task_1()