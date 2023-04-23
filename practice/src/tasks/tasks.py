import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def get_data():
    data = pd.read_csv('src/resources/dab.csv')

    for c in range(len((data.srednica2))):
        data.loc[c, ('srednica2')] = data.srednica2[c].replace(',', '.')

    X = np.concatenate((np.ones((len(data.values)-1, 1)), data.values[:-1, 0:2].astype(float)), axis=1)
    Y = data.values[:-1, 2].astype(float)
    
    return [X,Y]

def task_1():
    X, Y = get_data()

    beta = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:, 1], X[:, 2], Y)
    ax.plot(X[:, 1], X[:, 2], np.matmul(X, beta))
    
    rangeX = [min(X[:, 1]), max(X[:, 1])]
    rangeY = [min(X[:, 2]), max(X[:, 2])]
    rangeZ = np.matmul(np.vstack(([1, 1], rangeX, rangeY)).T, beta)
    ax.plot(rangeX, rangeY, rangeZ)

    ax.set_xlabel('srednica')
    ax.set_ylabel('srednica2')
    ax.set_zlabel('age')


    plt.show()



def task_2():
    from sklearn.linear_model import LinearRegression
    
    X, Y = get_data()

    lr = LinearRegression()
    lr.fit(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:, 1], X[:, 2], Y)
    ax.plot(X[:, 1], X[:, 2], lr.predict(X))

    plt.show()

def main():
    task_2()
