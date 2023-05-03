import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

matplotlib.use('TkAgg')

def task_1_2():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

    for i, n_samples in enumerate([1000, 200]):
        for j, noise in enumerate([.05, .3]):
            X, y = make_moons(n_samples=n_samples, noise=noise)

            svc = SVC(kernel='linear', C=1.0, random_state=0)
            svc.fit(X, y)

            OX, OY = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 100), np.linspace(min(X[:, 1]), max(X[:, 1]), 100))
            Z = [svc.predict([[OX[i,j], OY[i,j]] for j in range(100)]) for i in range(100)]
            
            axes[i,j].contourf(OX, OY, Z)
            axes[i,j].scatter(X[:,0], X[:,1], c=y, cmap='viridis')
            axes[i,j].set_title(f"n_samples={n_samples}, noise={noise}")


    plt.show()


def task_3():

    for (C, gamma) in [(.1, 1), (.1, 100), (100, 10)]:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

        for i, n_samples in enumerate([1000, 200]):
            for j, noise in enumerate([.05, .3]):
                X, y = make_moons(n_samples=n_samples, noise=noise)

                svc = SVC(kernel='rbf', C=C, gamma=gamma, random_state=0)
                svc.fit(X, y)

                OX, OY = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 100), np.linspace(min(X[:, 1]), max(X[:, 1]), 100))
                Z = [svc.predict([[OX[i,j], OY[i,j]] for j in range(100)]) for i in range(100)]
                
                axes[i,j].contourf(OX, OY, Z)
                axes[i,j].scatter(X[:,0], X[:,1], c=y, cmap='viridis')
                axes[i,j].set_title(f"n_samples={n_samples}, noise={noise}, C={C}, gamma={gamma}")

    plt.show()


def task_4():
    from sklearn import datasets

    # import Iris data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  
    y = iris.target

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    svc = SVC(kernel='rbf')
    gs = GridSearchCV(svc, { 'C': np.logspace(1,2, 10), 'gamma': np.logspace(-1, 2, 10) })
    gs.fit(Xt, y)

    svc = SVC(kernel='rbf', **gs.best_params_)
    svc.fit(Xt, y)

    OX, OY = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 100), np.linspace(min(X[:, 1]), max(X[:, 1]), 100))
    Z = [svc.predict(scaler.transform([[OX[i,j], OY[i,j]] for j in range(100)])) for i in range(100)]
    
    plt.contourf(OX, OY, Z)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def task_5():
    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets

    # import Iris data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  
    y = iris.target

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    logit = LogisticRegression()
    gs = GridSearchCV(logit, { 'C': 1/np.logspace(-1,2, 10) })
    gs.fit(Xt, y)

    logit = LogisticRegression(**gs.best_params_)
    logit.fit(Xt, y)

    OX, OY = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 100), np.linspace(min(X[:, 1]), max(X[:, 1]), 100))
    Z = [logit.predict(scaler.transform([[OX[i,j], OY[i,j]] for j in range(100)])) for i in range(100)]
    
    plt.contourf(OX, OY, Z)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

def main():
    task_5()
