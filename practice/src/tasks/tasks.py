import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.preprocessing as pr
import sklearn.model_selection as ms
import sklearn.neural_network as nn
import sklearn.metrics as m
import sklearn.linear_model as lm
import math
from imblearn.over_sampling import RandomOverSampler

matplotlib.use('TkAgg')

def task_1():
    # Data preperation
    data = pd.read_csv('./practice/src/resources/infmort.csv')
    print(data.head())

    regions = np.unique(data.iloc[:, 1])
    print('Regions: ', ', '.join(regions))

    X = pr.StandardScaler().fit_transform(data.iloc[:, 2:4])
    X = pd.DataFrame(X, columns=data.iloc[:, 2:4].columns)
    X[[f"region_{r}" for r in regions]] = data['region'].apply(lambda r: pd.Series((regions == r).astype(int)))

    y = data.iloc[:, 4]
    y = (y == y[0]).astype(int)

    X['y'] = y
    X.dropna(inplace=True)
    y = X['y']
    X = X.loc[:, X.columns != 'y']

    oversampler = RandomOverSampler()
    X1, y1 = oversampler.fit_resample(X, y)

    print(X.head())

    fig = plt.figure(figsize=(10, 6))

    for k in range(max(1, round(0.2 * len(X.columns))), round(1.5 * len(X.columns))):
        best_mlp = ms.GridSearchCV(nn.MLPClassifier(max_iter=100000, hidden_layer_sizes=(k,)), param_grid={
            'solver': ['lbfgs', 'sgd', 'adam'],
            'activation': ['tanh', 'relu', 'logistic'],
            'alpha': [0.001, 0.01, 0.1]
        })

        best_mlp.fit(X1, y1)

        yp = best_mlp.predict(X)

        CM = m.confusion_matrix(y, yp)
        print(f"\n\n====| k = {k} |====\n")
        print('Confusion matrix:\n', CM)
        print(f"f1 = {m.f1_score(y, yp)}")
        print(f"Percent of succesful predictions: {100 * ((CM[0,0] + CM[1,1])/np.sum(CM))}%")

        train_scores, valid_scores = ms.validation_curve(
            nn.MLPClassifier(max_iter=100000, **best_mlp.best_params_),
            X, y,
            param_name='hidden_layer_sizes',
            param_range=[(k,), (k,k), (k,k,k,), (k,k,k,k,), (k,k,k,k,k,)]
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)

        ax = fig.add_subplot(3,3,k)
        plt.title(f"Validation Curve for k={k}")
        plt.xlabel("Parameter Value")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        param_range = [1,2,3,4,5]
        ax.plot(param_range, train_mean, label="Training score", color="r")
        ax.plot(param_range, valid_mean, label="Cross-validation score", color="g")
        ax.fill_between(
            param_range,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color="r"
        )
        ax.fill_between(
            param_range,
            valid_mean - valid_std,
            valid_mean + valid_std,
            alpha=0.2,
            color="g"
        )
        plt.legend(loc="best")


    plt.show()

    print('Linera regression: ')

    clf = lm.LogisticRegression()
    clf.fit(X, y)

    yp = clf.predict(X)

    CM = m.confusion_matrix(y, yp)
    print('Confusion matrix:\n', CM)
    print(f"f1 = {m.f1_score(y, yp)}")
    print(f"Percent of succesful predictions: {100 * ((CM[0,0] + CM[1,1])/np.sum(CM))}%")

def main():
    task_1()
