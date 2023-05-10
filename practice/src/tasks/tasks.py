import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_iris, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

matplotlib.use('TkAgg')



def task_1():
    df = pd.read_csv('practice/src/resource/wdbc.data', header=None)

    print(df.head())

    scaler = StandardScaler()

    df[1]=df[1].replace('M', 1)
    df[1]=df[1].replace('B', 0)

    X = scaler.fit_transform(df.iloc[:, [2,31]])
    y = df.iloc[:, 1]

    def draw_accuracy_plot(kernel, ax=plt):
        nonlocal X, y

        svc = SVC(kernel=kernel, C=.1)
        
        train_sizes, train_scores, test_scores = learning_curve(
            svc, 
            X, 
            y, 
            train_sizes=np.linspace(0.1, 1, 10), 
            cv=5, 
            scoring='accuracy'
        )

        mean_train = np.mean(train_scores, axis=1)
        mean_test = np.mean(test_scores, axis=1)
        std_train = np.std(train_scores, axis=1)
        std_test = np.std(test_scores, axis=1)

        ax.plot(train_sizes, mean_train, label='Training score', color='r')
        ax.plot(train_sizes, mean_test, label='Test score', color='b')
        ax.fill_between(train_sizes, mean_train - std_train, mean_train + std_train, alpha=0.1, color='r')
        ax.fill_between(train_sizes, mean_test - std_test, mean_test + std_test, alpha=0.1, color='b')

        plt.title(kernel)
        ax.legend()


    fig = plt.figure()
    draw_accuracy_plot('linear', fig.add_subplot(211))
    draw_accuracy_plot('rbf', fig.add_subplot(212))
    plt.show()


def task_2():
    data = load_iris()
    X = data.data
    y = data.target

    knn = KNeighborsClassifier()
    K = list(range(1, 29))

    train_scores, test_scores = validation_curve(
        knn, X, y, param_name='n_neighbors', param_range=K, cv=10
    )

    mean_train = np.mean(train_scores, axis=1)
    mean_test = np.mean(test_scores, axis=1)

    std_train = np.std(train_scores, axis=1)
    std_test = np.std(test_scores, axis=1)

    plt.plot(K, mean_train)
    plt.plot(K, mean_test)

    plt.fill_between(K, mean_train - std_train, mean_train + std_train, alpha=0.2)
    plt.fill_between(K, mean_test - std_test, mean_test + std_test, alpha=0.2)

    plt.legend()
    plt.show()


def task_3():
    X, y = make_moons(n_samples=200, noise=.3)

    knn = KNeighborsClassifier()
    K = list(range(1, 39))

    train_scores, test_scores = validation_curve(
        knn, X, y, param_name='n_neighbors', param_range=K, cv=10
    )

    mean_train = np.mean(train_scores, axis=1)
    mean_test = np.mean(test_scores, axis=1)

    std_train = np.std(train_scores, axis=1)
    std_test = np.std(test_scores, axis=1)

    plt.plot(K, mean_train)
    plt.plot(K, mean_test)

    plt.fill_between(K, mean_train - std_train, mean_train + std_train, alpha=0.2)
    plt.fill_between(K, mean_test - std_test, mean_test + std_test, alpha=0.2)

    plt.legend()
    plt.show()


def task_4():
    X, y = make_moons(n_samples=200, noise=.3)

    svm=SVC(kernel='rbf', random_state=0)
    C = np.linspace(100, 0.5, 100)

    train_scores, test_scores = validation_curve(
        svm, X, y, param_name='C', param_range=C, cv=6, scoring='accuracy'
    )

    mean_train = np.mean(train_scores, axis=1)
    mean_test = np.mean(test_scores, axis=1)

    std_train = np.std(train_scores, axis=1)
    std_test = np.std(test_scores, axis=1)

    plt.plot(C, mean_train, label='train')
    plt.plot(C, mean_test, label='test')

    plt.fill_between(C, mean_train - std_train, mean_train + std_train, alpha=0.2)
    plt.fill_between(C, mean_test - std_test, mean_test + std_test, alpha=0.2)

    plt.legend()
    plt.show()


def task_5():
    n_samples, n_features = 300, 50
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=1
    )

    svm=SVC(kernel='rbf', random_state=0)
    C = np.linspace(100, 0.5, 100)

    train_scores, test_scores = validation_curve(
        svm, X, y, param_name='C', param_range=C, cv=6, scoring='accuracy'
    )

    mean_train = np.mean(train_scores, axis=1)
    mean_test = np.mean(test_scores, axis=1)

    std_train = np.std(train_scores, axis=1)
    std_test = np.std(test_scores, axis=1)

    plt.plot(C, mean_train, label='train')
    plt.plot(C, mean_test, label='test')

    plt.fill_between(C, mean_train - std_train, mean_train + std_train, alpha=0.2)
    plt.fill_between(C, mean_test - std_test, mean_test + std_test, alpha=0.2)

    plt.legend()
    plt.show()


def main():
    task_5()