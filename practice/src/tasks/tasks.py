import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pr
import numpy as np
import pandas as pd
import math
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV

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

def task_2():
    # (X^T*X + lambda*I)*beta = X^T*y
    # find beta:
    lamb = 1
    n = 3

    x = np.linspace(0,2*np.pi,40)
    y = np.sin(x)+np.random.normal(0,0.4,40)

    pf = pr.PolynomialFeatures(n)

    X = pf.fit_transform(x.reshape(-1, 1))
    I = np.identity(n+1)
    I[0,0] = 0
    A = lambda l: (np.matmul(X.T, X) + l * I)
    b = np.matmul(X.T, y)

    beta = np.linalg.solve(A(lamb), b)

    rr = lm.Ridge(alpha=lamb)
    rr.fit(X, y)

    print(np.sum((beta - rr.coef_)**2))
    print(beta, rr.coef_)

    plt.scatter(x, y)
    plt.plot(x, rr.predict(X))
    plt.plot(x, [np.matmul(a, beta) for a in X])
    plt.show()


def task_3():
    N = 3000

    df1 = lambda n: 5 * pd.DataFrame(np.random.randn(n, 3), columns=['A', 'B','C'])
    df2 = lambda n: 10 + 10 * pd.DataFrame(np.random.randn(n, 3), columns=['A', 'B','C'])

    Data = np.concatenate((df1(N), df2(N)), axis=0)
    Data[:,2] = 1
    Data[0:N,2] = 0

    # Train on X[:, 0:2]
    log_reg1 = lm.LogisticRegression()
    log_reg1.fit(Data[:, 0:2], Data[:, 2])

    print('log_reg1 R2 score:', log_reg1.score(Data[:, :-1], Data[:, -1:]))

    # Train on X[:, 1] and powers of X[:, 0] from 0 to 4
    Data2 = np.hstack((pr.PolynomialFeatures(4).fit_transform(Data[:, :1]), Data[:, 1:]))
    log_reg2 = lm.LogisticRegression()
    log_reg2.fit(Data2[:, :-1], Data2[:, -1:])

    print('log_reg2 R2 score:', log_reg2.score(Data2[:, :-1], Data2[:, -1:]))

    # First train but with penalty functions and lamda \in [1, 10]
    log_reg3 = lm.LogisticRegression(penalty='l2', solver='liblinear', C=1/(1 * len(Data2[:, :-1])))
    log_reg3.fit(Data[:, 0:2], Data[:, 2])
    print('log_reg3 R2 score:', log_reg3.score(Data[:, 0:2], Data[:, 2]))

    log_reg4 = lm.LogisticRegression(penalty='l2', solver='liblinear', C=1/(10 * len(Data2[:, :-1])))
    log_reg4.fit(Data[:, 0:2], Data[:, 2])
    print('log_reg4 R2 score:', log_reg4.score(Data[:, 0:2], Data[:, 2]))

    # Second train but with penalty functions and lamda \in [1, 10]
    log_reg5 = lm.LogisticRegression(penalty='l2', solver='liblinear', C=1/(1 * len(Data2[:, :-1])))
    log_reg5.fit(Data2[:, :-1], Data2[:, -1:])
    print('log_reg5 R2 score:', log_reg5.score(Data2[:, :-1], Data2[:, -1:]))

    log_reg6 = lm.LogisticRegression(penalty='l2', solver='liblinear', C=1/(10 * len(Data2[:, :-1])))
    log_reg6.fit(Data2[:, :-1], Data2[:, -1:])
    print('log_reg6 R2 score:', log_reg6.score(Data2[:, :-1], Data2[:, -1:]))

    # Printing scores:
    print(classification_report(Data[:, -1:], log_reg1.predict(Data[:, :-1])))
    print(confusion_matrix(Data[:, -1:], log_reg1.predict(Data[:, :-1])))
    print(classification_report(Data2[:, -1:], log_reg2.predict(Data2[:, :-1])))
    print(confusion_matrix(Data2[:, -1:], log_reg2.predict(Data2[:, :-1])))
    print(classification_report(Data[:, -1:], log_reg3.predict(Data[:, :-1])))
    print(confusion_matrix(Data[:, -1:], log_reg3.predict(Data[:, :-1])))
    print(classification_report(Data[:, -1:], log_reg4.predict(Data[:, :-1])))
    print(confusion_matrix(Data[:, -1:], log_reg4.predict(Data[:, :-1])))
    print(classification_report(Data2[:, -1:], log_reg5.predict(Data2[:, :-1])))
    print(confusion_matrix(Data2[:, -1:], log_reg5.predict(Data2[:, :-1])))
    print(classification_report(Data2[:, -1:], log_reg6.predict(Data2[:, :-1])))
    print(confusion_matrix(Data2[:, -1:], log_reg6.predict(Data2[:, :-1])))

    # Prepare before plotting decision boundaries

    ## c + bx = 0
    ## bx = -c
    ## b1*x1 = -c - b'*x'
    ## x1 = -(c + b'*x') / b1
    x1 = lambda x, lr: -(lr.intercept_[0] + np.dot(np.array(x), lr.coef_[0][:-1])) / lr.coef_[0][-1]

    # Plot decision boundaries for log_reg1 and log_reg2
    y_lims = 1.1 * np.array([min(Data[:, 1]), max(Data[:, 1])])
    x_lims = 1.1 * np.array([min(Data[:, 0]), max(Data[:, 0])])
    OX = np.linspace(*x_lims, 100).reshape(-1,1)
    OY1 = np.array([x1(x, log_reg1) for x in OX])
    OY2 = np.array([x1(x, log_reg2) for x in pr.PolynomialFeatures(4).fit_transform(OX)])
    OY3 = np.array([x1(x, log_reg3) for x in OX])
    OY4 = np.array([x1(x, log_reg4) for x in OX])
    OY5 = np.array([x1(x, log_reg5) for x in pr.PolynomialFeatures(4).fit_transform(OX)])
    OY6 = np.array([x1(x, log_reg6) for x in pr.PolynomialFeatures(4).fit_transform(OX)])
    
    # Ploting countourf 
    ## OY = np.linspace(*y_lims, 100).reshape(-1,1)
    ## X,Y = np.meshgrid(OX, OY)
    ## Z1 = [log_reg1.predict([[X[i,j], Y[i,j]] for j in range(100)]) for i in range(100)]
    ## Z2 = [log_reg2.predict([[*(lambda x: [x**i for i in range(5)])(X[i,j]), Y[i,j]] for j in range(100)]) for i in range(100)]
    ## plt.contourf(X, Y, Z2)

    plt.plot(OX, OY1, label='t1 no penalty')
    plt.plot(OX, OY2, label='t2 no penalty')
    plt.plot(OX, OY3, label='t1, lambda=1')
    plt.plot(OX, OY4, label='t1, lambda=10')
    plt.plot(OX, OY5, label='t2, lambda=1')
    plt.plot(OX, OY6, label='t2, lambda=10')
    plt.ylim(*y_lims)
    plt.xlim(*x_lims)

    s = 'b' * N + 'r' * N
    plt.scatter(Data[:,0], Data[:,1], s=5, alpha=1, c=list(s))
    plt.legend()
    plt.show()


def task_4():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    def with_regularization(X, y, ax=plt):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        scaler = pr.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = GridSearchCV(lm.LogisticRegression(penalty='l2', solver='liblinear'), {'C': [1/(i * len(X_train)) for i in range(1, 10)]}, cv=5)
        clf.fit(X_train_scaled, y_train)
        best_C = clf.best_params_['C']

        lr = lm.LogisticRegression(penalty='l2', C=best_C)
        lr.fit(X_train_scaled, y_train)
        
        print(classification_report(y_test, lr.predict(X_test_scaled)))
        print('lambda='+str(1/(best_C*len(X_train)))+'; R2='+str(lr.score(X_test_scaled, y_test))+';\n', confusion_matrix(y_test, lr.predict(X_test_scaled)))

        OX = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        OY = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
        Xp, Yp = np.meshgrid(OX, OY)
        Zp = [lr.predict(scaler.transform([[Xp[i,j], Yp[i,j]] for j in range(100)])) for i in range(100)]

        ax.contourf(Xp, Yp, Zp)
        ax.scatter(X[:, 0], X[:, 1], c=y)

    def without_refularization(X, y, ax=plt):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        scaler = pr.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = lm.LogisticRegression()
        lr.fit(X_train_scaled, y_train)
        
        print(classification_report(y_test, lr.predict(X_test_scaled)))
        print('R2='+str(lr.score(X_test_scaled, y_test))+';\n', confusion_matrix(y_test, lr.predict(X_test_scaled)))

        OX = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        OY = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
        Xp, Yp = np.meshgrid(OX, OY)
        Zp = [lr.predict(scaler.transform([[Xp[i,j], Yp[i,j]] for j in range(100)])) for i in range(100)]

        ax.contourf(Xp, Yp, Zp)
        ax.scatter(X[:, 0], X[:, 1], c=y)


    fig = plt.figure()
    with_regularization(X, y, fig.add_subplot(211))
    without_refularization(X, y, fig.add_subplot(212))
    plt.show()

def main():
    task_3()