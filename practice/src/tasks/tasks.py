import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.metrics import pairwise_distances

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

def task_2():
    iris = datasets.load_iris()
    X = iris.data[:,:2]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the range of clusters to explore
    k_values = range(2, 11)

    inertias = []

    for k in k_values:
        model = KMeans(n_clusters=k)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    # Plotting the sum of squared errors
    plt.figure(figsize=(10, 5))

    # Sum of squared errors plot
    plt.subplot(121)
    plt.plot(k_values, inertias, 'o-')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared errors (Inertia)')

    # Logarithm of the sum of squared errors plot
    plt.subplot(122)
    plt.plot(k_values, np.log(inertias), 'o-')
    plt.title('Elbow method with logarithm')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Logarithm of the Sum of squared errors (Inertia)')

    plt.tight_layout()
    plt.show()

def task_2_():
    iris = datasets.load_iris()
    X = iris.data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Define the range of clusters to explore
    k_values = range(1, 21)

    # For storing Wk values
    Wk_values = []

    for k in k_values:
        # Cluster the observed data and compute Wk
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        Wk_values.append(kmeans.inertia_)

    # Generate B reference datasets and compute the estimated gap statistic
    B = 10
    gap_values = []

    for k in k_values:
        ref_Wk_values = []
        
        for b in range(B):
            # Generate a reference dataset
            random_data = np.random.rand(*X.shape)
            # Cluster it and compute Wk
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(random_data)
            ref_Wk = kmeans.inertia_
            ref_Wk_values.append(ref_Wk)
        
        # Compute the gap statistic
        gap = np.mean(np.log(ref_Wk_values) - np.log(Wk_values[k-1]))
        gap_values.append(gap)

    # Plotting gap values
    plt.plot(k_values, gap_values, 'o-')
    plt.title('Gap Statistic')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.show()

def main():
    task_2_()