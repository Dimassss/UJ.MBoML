from sklearn.model_selection import train_test_split
from src.tasks.task_1 import nearest
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def task_4():
    N = 1000
    K = 5
    S = 100

    df1 = 2 * pd.DataFrame(np.random.randn(N, 3), columns=['A', 'B', 'C'])
    df2 = 20 + 10 * pd.DataFrame(np.random.randn(N, 3), columns=['A', 'B', 'C'])
    data = np.concatenate((df1, df2), axis=0)
    data[:, 2] = 1
    data[0:N, 2] = 0

    X = data[:, 0:2]
    Y = data[:, 2]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)
    
    neigh = KNeighborsClassifier(n_neighbors=4)
    neigh.fit(X_train, Y_train)

    OX = np.linspace(min(data[:, 0]), max(data[:, 0]), S)
    OY = np.linspace(min(data[:, 1]), max(data[:, 1]), S)
    OX, OY = np.meshgrid(OX, OY)
    Z = [[neigh.predict([[OX[i,j], OY[i,j]]]) for i in range(S)] for j in range(S)]
    Z = np.array(Z).reshape(S, S)

    plt.contourf(OX, OY, Z)
    plt.colorbar()
    
    s = ['#FF0000' if y == 0 else '#00FF00' for y in Y_train]
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha=1, c=s)

    print(confusion_matrix(Y_test, [neigh.predict([x]) for x in X_test]))

    plt.show()

    
