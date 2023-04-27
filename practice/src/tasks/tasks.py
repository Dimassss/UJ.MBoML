import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pr
import numpy as np
import math

matplotlib.use('TkAgg')


def task_1():
    x=np.linspace(0,2*np.pi,40)
    y=np.sin(x)+np.random.normal(0,0.4,40)

    def draw_poly_reg(X,Y, n, ax=plt):
        pf = pr.PolynomialFeatures(n)
        X_poly = pf.fit_transform(X)

        lr = lm.LinearRegression()
        lr.fit(X_poly, Y)

        x = np.linspace(min(X), max(X), 1000)
        y = lr.predict(pf.transform(x))
        ax.plot(x, y)

    def draw_ridge_regression(X, Y, n, lamb, ax=plt):
        pf = pr.PolynomialFeatures(n)
        X_poly = pf.fit_transform(X)
        
        rr = lm.Ridge(alpha=lamb)
        rr.fit(X_poly, Y)

        x = np.linspace(min(X), max(X), 1000)
        y = rr.predict(pf.transform(x))
        ax.plot(x, y)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    ax1.scatter(x,y)
    ax2.scatter(x,y)
    
    draw_ridge_regression(x.reshape(-1, 1), y.reshape(-1, 1), 17, 1, ax2)
    draw_ridge_regression(x.reshape(-1, 1), y.reshape(-1, 1), 17, 100, ax2)
    draw_poly_reg(x.reshape(-1, 1), y.reshape(-1, 1), 3, ax1)
    draw_poly_reg(x.reshape(-1, 1), y.reshape(-1, 1), 17, ax1)
    
    plt.show()


def main():
    task_1()
