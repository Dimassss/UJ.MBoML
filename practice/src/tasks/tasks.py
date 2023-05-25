import numpy as np
import matplotlib
from  sklearn.neural_network import MLPClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def task_1():
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


def task_2(n):
    # n = number of neurons in hidden layer
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

    W1 = 2*np.random.random((3,n)) - 1
    W2 = 2*np.random.random((n,2)) - 1

    for i in range(1000):
        l0 = X
        l1 = nonlin(np.dot(l0, W1))
        l2 = nonlin(np.dot(l1, W2))

        l2_error = y - l2
        l2_delta = l2_error * nonlin(l2, True)

        l1_error = l2_delta.dot(W2.T)
        l1_delta = l1_error * nonlin(l1, True)

        W1 += l0.T.dot(l1_delta)
        W2 += l1.T.dot(l2_delta)


    l0 = X
    l1 = nonlin(np.dot(l0, W1))
    l2 = nonlin(np.dot(l1, W2))

    print(np.mean(np.abs(l2 - y)))

def task_3():
    """
    http://playground.tensorflow.org/#activation=tanh&regularization=L2&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.001&regularizationRate=0.1&noise=0&networkShape=1&seed=0.23623&showTestData=false&discretize=false&percTrainData=80&x=false&y=false&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
    http://playground.tensorflow.org/#activation=tanh&regularization=L2&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.001&regularizationRate=0.1&noise=0&networkShape=1&seed=0.85487&showTestData=false&discretize=false&percTrainData=80&x=false&y=false&xTimesY=true&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
    http://playground.tensorflow.org/#activation=sigmoid&regularization=L2&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.001&regularizationRate=0.1&noise=0&networkShape=1&seed=0.10434&showTestData=false&discretize=false&percTrainData=80&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false


    """

def task_4():
    def train(n=0, ax=plt):
        iris = datasets.load_iris()
        X = iris.data[:, :2]
        y = iris.target

        mlp=MLPClassifier(hidden_layer_sizes=((n,) if n > 0 else []),activation='logistic',max_iter=5000)
        mlp.fit(X, y)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        ax.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("NN for "+str(n)+" neurons")

    train(0, plt.subplot(2, 2, 1))
    train(2, plt.subplot(2, 2, 2))
    train(4, plt.subplot(2, 2, 3))
    train(10, plt.subplot(2, 2, 4))

    plt.show()

def main():
    task_4()
