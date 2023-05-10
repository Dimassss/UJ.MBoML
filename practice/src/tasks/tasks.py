import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import matplotlib

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
            train_sizes=np.linspace(0.1, 0.9, 9), 
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


def main():
    task_1()