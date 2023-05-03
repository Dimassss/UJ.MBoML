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
import seaborn as sns
import math
import itertools

matplotlib.use('TkAgg')

def data_wrangling(df):
    # Filter only useful columns
    print(df.columns)
    filter_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    print('Filter columns: ', filter_columns)
    df = df[filter_columns]
    print(df.head(7))

    # Categorical values to integer
    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1
    print(df)

    # preprocessing NA values
    fig_nan, axes_nan = plt.subplots(nrows=2, ncols=1, figsize=(2,2))
    sns.heatmap(df.isna(), cmap='viridis', ax=axes_nan[0])
    #plt.show()

    # as we can see from the plot, null values are only in age and survived columns
    # firstly I will use values from dataset to predict age, so I can rid of null values from age columns
    # then I will remove records with survived columns equal to null, because we can do nothing with that

    # prepare X and and y value to set Age column
    df_age = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age']]
    df_train = df_age[[not a for a in np.isnan(df_age['Age'])]]
    X = df_train[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]
    y = df_train['Age']

    # Lets take a look which dependece is between Age and other columns.It will help us to create better model.
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(2,2))

    for i, cols in enumerate([['Pclass', 'Sex'], ['SibSp', 'Parch']]):
        for j, col in enumerate(cols):
            ax = axes[i][j]
            ax.set_xlabel('Age')
            ax.set_ylabel('People count')

            for v in set(X[col]):   
                ax.hist(y[X[col] == v], alpha=0.4, bins=10, label=col+'='+str(v))


            ax.legend()

    # Ploting Fare - Age graph
    ax = axes[2][0]
    age_bins = [10 * np.array([i, i+1]) for i in range(10)]
    d = [X['Fare'][(age_bin[0] <= y) & (y < age_bin[1])] for age_bin in age_bins]
    ax.boxplot(d)
    ax.set_xticks(list(range(1, len(age_bins)+1)), [str(ab[0])+'-'+str(ab[1]) for ab in age_bins], rotation=70)

    # plt.show()

    # From all of this plots we see that Fare column is not very helpful to predict age. 
    X = X[['Pclass', 'Sex', 'SibSp', 'Parch']]

    # Now we can try build a model which predict Age and test it
    def generate_polynoms(n, X):
        X_new_cols = []
        cn = len(X.values[0])
        rn = len(X.values)
        columns = list(range(cn)) + [-1 for i in range(cn)]

        for col_comb in set(itertools.combinations(columns, n)):
            new_col = np.array([1 for i in range(rn)]).reshape(-1, 1)

            for c in col_comb:
                if c == -1:
                    continue
                else:
                    new_col = new_col * X.values[:, c:(c+1)]

            if len(X_new_cols) > 0:
                X_new_cols = np.hstack((X_new_cols, new_col))
            else:
                X_new_cols = new_col

        Y = pd.DataFrame(np.hstack([X, pd.DataFrame(X_new_cols)]))
        x_cols = list(X.columns)
        y_cols = list(Y.columns)
        Y.columns = x_cols + [str(i) for i in range(len(y_cols) - len(x_cols))]

        return Y

    # X_poly = generate_polynoms(2, X)
    # ridge = Ridge()
    # gs = GridSearchCV(ridge, {'alpha': np.logspace(-1, 2, 10)}, cv=10, scoring='neg_mean_squared_error')
    # gs.fit(X_poly.values, y)

    # ridge = Ridge(alpha=gs.best_params_['alpha'])
    # ridge.fit(X_poly, y)

    # print(ridge.score(X_poly, y), gs.best_params_['alpha'])
    # print(classification_report(y, ridge.predict(X)))

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=20)
    
    y_categorized = np.array([*y])
    for cbin in [[0, 10]] + [np.logspace(1, 2, 6)[i:(i+2)] for i in range(5)]:
        y_categorized[(y >= cbin[0]) & (y < cbin[1])] = int(np.mean(cbin))

    rfc.fit(X, y_categorized)

    print(rfc.score(X, y_categorized))
    print(set(y_categorized))

    # As we cann there is 50% chance that we predict age correcly.
    # Amount of records with Age!=NaN is much bigget that amount of records with Age=NaN.
    # So we can use such model to fulfil empty values of Age column

    for i in range(len(df.values)):
        if math.isnan(df.loc[i, 'Age']):
            df.loc[i, 'Age'] = rfc.predict(df.loc[i:(i+1), ['Pclass', 'Sex', 'SibSp', 'Parch']])[0]

    sns.heatmap(df.isna(), cmap='viridis', ax=axes_nan[1])

    plt.show()
    # As we can see now, all the columns are filled, so we can try to train our model in order to predicr survive column
    # So out data is clean and prepared now

    return df

def processing(df):
    pass

def main():
    df = pd.read_csv('practice/resources/full.csv')

    data_wrangling(df)