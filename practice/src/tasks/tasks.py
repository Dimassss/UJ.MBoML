import numpy as np
import matplotlib

matplotlib.use('TkAgg')


def task_1():

    # sigmoid function
    def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    X = np.array([  
        [1,1,1],
        [1,0,1],
        [0,1,1],
        [0,0,1],
        [0,0,0],
        [1,1,0]
    ])

    y = np.array([  
        [1,1],
        [1,1],
        [1,0],
        [0,0],
        [0,0],
        [1,1]
    ])

    np.random.seed(1)
    W1 = np.array([2*np.random.random((3,1)) - 1 for i in range(2)]).T

    for i in range(1000):
        for j in range(len(X)):
            l0 = X[j,:]
            print()
            l1 = nonlin(np.dot(l0, W1))
            
            l1_err = y[j,:] - l1
            l1_delta = l1_err * nonlin(l1, True)

            W1 += np.dot(l0.reshape(-1,1), l1_delta)

    err = lambda k: np.linalg.norm(nonlin(np.dot(X[k,:], W1)) - y[k])
    print(np.mean([err(k) for k in range(len(y))]))


def main():
    task_1()
