import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

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

def main():
    task_1()
