import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

matplotlib.use('TkAgg')

def minmax(min_v, max_v, v):
    return min(max(min_v, v), max_v)

def task_1():
    N=1000
    df1 = 2*pd.DataFrame(np.random.randn(N, 3), columns=['A', 'B','C'])
    df2 = 20+10*pd.DataFrame(np.random.randn(N, 3), columns=['A', 'B','C'])

    Data=np.concatenate((df1, df2), axis=0)
    Data[:,2]=1
    Data[0:N,2]=0

    # my code

    ## split data into train and test blocks
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Data[:, 0:2], Data[:, 2], test_size=0.2)

    # Trains

    ## Train KNN
    from sklearn.neighbors import KNeighborsClassifier
    M1 = KNeighborsClassifier(n_neighbors=3)
    M1.fit(X_train, y_train)
    M1_predict = lambda points: M1.predict(points)

    ## Train linear regresisons
    from sklearn.linear_model import LinearRegression
    M2 = LinearRegression()
    M2.fit(X_train, y_train)
    M2_predict = lambda points: [(0 if y < 0.5 else 1) for y in M2.predict(points)]

    ## Train logistic regression
    from sklearn.linear_model import LogisticRegression
    M3 = LogisticRegression()
    M3.fit(X_train, y_train)
    M3_predict = lambda points: M3.predict(points)

    # Test models

    print('Test model M1')
    print('R2 for test data:', M1.score(X_test, y_test))

    print('\nTest model M2')
    y_test_values = M2_predict(X_test)
    SSR2 = np.sum((y_test - y_test_values)**2)
    SST2 = np.sum((y_test - np.mean(y_test))**2)
    R2 = 1 - SSR2 / SST2
    print('R2 for test data:', R2)
    
    print('\nTest model M3')
    print('R2 for test data:', M3.score(X_test, y_test))
    print('Intercept:', M3.intercept_)
    print('Coefficient:', M3.coef_)
    
    # Illustrate models

    # M1
    X_span = np.linspace(min(Data[:,0]), max(Data[:,0]), 100)
    Y_span = np.linspace(min(Data[:,1]), max(Data[:,1]), 100)
    OX, OY = np.meshgrid(X_span, Y_span)
    Z = [[M1_predict([[OX[i,j], OY[i,j]]])[0] for i in range(100)] for j in range(100)]
    plt.contourf(OX, OY, Z)

    # M2
    b1, b2 = M2.coef_
    b0 = M2.intercept_
    f2 = lambda x: (0.5 - b0 - b1*x) / b2
    f2_inv = lambda y: -(b2*y - 0.5 + b0) / b1
    
    X_range = [min(Data[:,0]), max(Data[:,0])]
    Y_range = [min(Data[:,1]), max(Data[:,1])]
    Yf_range = sorted([f2(x) for x in X_range])

    if Y_range[0] <= Yf_range[1] and Y_range[1] >= Yf_range[0]:
        OX = [minmax(X_range[0], X_range[1], f2_inv(y)) for y in Y_range]
        OY = [f2(x) for x in OX]
        plt.plot(OX, OY)

    # M3
    b1, b2 = M3.coef_[0]
    b0 = M3.intercept_[0]
    f3 = lambda x: -(b0 + b1 * x) / b2
    f3_inv = lambda y: -(b0 + y*b2) / b1

    X_range = [min(Data[:,0]), max(Data[:,0])]
    Y_range = [min(Data[:,1]), max(Data[:,1])]
    Yf_range = sorted([f3(x) for x in X_range])

    if Y_range[0] <= Yf_range[1] and Y_range[1] >= Yf_range[0]:
        OX = [minmax(X_range[0], X_range[1], f3_inv(y)) for y in Y_range]
        OY = [f3(x) for x in OX]
        plt.plot(OX, OY)

    # Preparations to draw plot
    s='b'*N+'r'*N
    plt.scatter(Data[:,0],Data[:,1], s=5,alpha=1,c=list(s))

    plt.show()


def main():
    task_1()
