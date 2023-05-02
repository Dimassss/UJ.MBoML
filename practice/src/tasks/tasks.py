import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV
import sklearn.preprocessing as pr
import sklearn.model_selection as ms
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

matplotlib.use('TkAgg')

def task_1():
    x=np.linspace(0,2*np.pi,40)
    x=np.append(x,10)
    y=np.sin(x)+np.random.normal(0,0.4,len(x))

    # Ridge
    ridge = Ridge()
    gs = GridSearchCV(ridge, {'alpha': np.logspace(-1, 3, 10)}, cv=10, scoring='neg_mean_squared_error')
    gs.fit(x.reshape(-1,1), y)
    
    alpha = gs.best_params_['alpha']
    ridge = Ridge(alpha=alpha)
    ridge.fit(x.reshape(-1,1), y)
    print('Ridge alpha =', alpha)
    print('Ridge Score: ', ridge.score(x.reshape(-1, 1), y))

    # LASSO
    lasso = Lasso()
    gs = GridSearchCV(lasso, {'alpha': np.logspace(-1, 3, 10)}, cv=10, scoring='neg_mean_squared_error')
    gs.fit(x.reshape(-1,1), y)
    
    alpha = gs.best_params_['alpha']
    lasso = Lasso(alpha=alpha)
    lasso.fit(x.reshape(-1,1), y)
    print('LASSO alpha =', alpha)
    print('LASSO Score: ', lasso.score(x.reshape(-1, 1), y))

    # Plotting

    ox = np.linspace(min(x), max(x), 100)
    plt.plot(ox, ridge.predict(ox.reshape(-1,1)), label='Ridge')
    plt.plot(ox, lasso.predict(ox.reshape(-1,1)), label='Lasso')

    plt.scatter(x,y)

    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.legend()
    plt.show()

def task_2():
    x=np.linspace(0,2*np.pi,40)
    y=np.sin(x)+np.random.normal(0,0.4,40)

    pf = pr.PolynomialFeatures(17)

    x_train, x_test, y_train, y_test = ms.train_test_split(pf.fit_transform(x.reshape(-1,1)),y)


    lasso = Lasso()
    gs = GridSearchCV(lasso, {'alpha': np.logspace(-2,2, 50)}, cv=6, scoring='neg_mean_absolute_error')
    gs.fit(x_train, y_train)
    
    lasso = Lasso(alpha=gs.best_params_['alpha'])
    lasso.fit(x_train, y_train)

    print("Score:", lasso.score(x_test, y_test))
    print("Best alpha: ", gs.best_params_['alpha'])

    ox = np.linspace(min(x), max(x), 100)
    plt.plot(ox, lasso.predict(pf.fit_transform(ox.reshape(-1,1))))

    plt.scatter(x,y)
    plt.show()

def task_3():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features
    y = iris.target

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_size=0.7)

    lr = LogisticRegression(penalty='l1', solver='liblinear')
    gs = GridSearchCV(lr, {'C': 1 / (len(X_train)*np.logspace(-2, 3, 100))}, cv=8, scoring='neg_mean_absolute_error')
    gs.fit(X_train, y_train)

    lr1 = LogisticRegression(penalty='l1', solver='liblinear', C=gs.best_params_['C'])
    lr1.fit(X_train, y_train)

    lr2 = LogisticRegression()
    lr2.fit(X_train, y_train)

    print(confusion_matrix(y_test, lr1.predict(X_test)))
    print(confusion_matrix(y_test, lr2.predict(X_test)))

    Xp, Yp = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 100), np.linspace(min(X[:, 1]), max(X[:, 1]), 100))
    Z1 = [lr1.predict([[Xp[i,j], Yp[i,j]] for j in range(100)]) for i in range(100)]
    Z2 = np.array([lr2.predict([[Xp[i,j], Yp[i,j]] for j in range(100)]) for i in range(100)])

    plt.contour(Xp, Yp, Z1)
    plt.contour(Xp, Yp, Z2)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

def main():
    task_3()
