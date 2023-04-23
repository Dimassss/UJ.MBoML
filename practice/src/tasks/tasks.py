import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def get_data():
    data = pd.read_csv('src/resources/dab.csv')

    for c in range(len((data.srednica2))):
        data.loc[c, ('srednica2')] = data.srednica2[c].replace(',', '.')

    X = np.concatenate((np.ones((len(data.values)-1, 1)), data.values[:-1, 0:2].astype(float)), axis=1)
    Y = data.values[:-1, 2].astype(float)
    
    return [X,Y]

def task_1():
    X, Y = get_data()

    beta = np.linalg.solve(np.matmul(X.T, X), np.matmul(X.T, Y))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:, 1], X[:, 2], Y)
    ax.plot(X[:, 1], X[:, 2], np.matmul(X, beta))
    
    rangeX = [min(X[:, 1]), max(X[:, 1])]
    rangeY = [min(X[:, 2]), max(X[:, 2])]
    rangeZ = np.matmul(np.vstack(([1, 1], rangeX, rangeY)).T, beta)
    ax.plot(rangeX, rangeY, rangeZ)

    ax.set_xlabel('srednica')
    ax.set_ylabel('srednica2')
    ax.set_zlabel('age')


    plt.show()



def task_2():
    from sklearn.linear_model import LinearRegression
    
    X, Y = get_data()

    lr = LinearRegression()
    lr.fit(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X[:, 1], X[:, 2], Y)
    ax.plot(X[:, 1], X[:, 2], lr.predict(X))

    plt.show()


def task_3():
    from sklearn.linear_model import LinearRegression
    N = 1000
    n = 1000

    # Generate x values from a uniform distribution between -1 and 1
    x = np.random.uniform(-1, 1, n)

    # Generate the error term from a standard normal distribution
    e = np.random.normal(0, 1, n)

    # Generate y values
    y = x**2 + e

    # Split the dataset into two parts based on the sign of x
    x_neg = x[x < 0]
    y_neg = y[x < 0]
    x_pos = x[x >= 0]
    y_pos = y[x >= 0]

    # Train a linear regression model on the first part of the dataset (x < 0)
    lr = LinearRegression()
    lr.fit(x_neg.reshape(-1, 1), y_neg)

    # Print the coefficients and R^2 score
    print("Intercept:", lr.intercept_)
    print("Coefficient:", lr.coef_)
    print("R^2 score:", lr.score(x_neg.reshape(-1, 1), y_neg))

    # Calculate the residuals on the test set (x > 0)
    residuals = y_pos - lr.predict(x_pos.reshape(-1, 1))

    # Plot the histogram of residuals
    plt.hist(residuals, bins=20)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    # Calculate the mean squared error (MSE) on the training set (x < 0)
    mse_train = np.mean((y_neg - lr.predict(x_neg.reshape(-1, 1)))**2)
    print("MSE on training set:", mse_train)

    # Calculate the mean squared error (MSE) on the test set (x > 0)
    mse_test = np.mean((y_pos - lr.predict(x_pos.reshape(-1, 1)))**2)
    print("MSE on test set:", mse_test)


def task_4():
    N = 5000
    
    x = np.random.uniform(-1, 1, N);
    e = 0.2 * np.random.normal(0,1, N)
    y = x**2 + e

    x_neg = x[x < 0].reshape(-1,1)
    x_pos = x[x >= 0].reshape(-1,1)
    y_neg = y[x < 0]
    y_pos = y[x >= 0]

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    pf = PolynomialFeatures(2)
    x2_pos = pf.fit_transform(x_pos)
    x2_neg = pf.fit_transform(x_neg)

    lr = LinearRegression()
    lr.fit(x2_pos, y_pos)

    print('Intercept:', lr.intercept_)
    print('Coefficient:', lr.coef_)
    print('R^2 score: ', lr.score(x2_neg, y_neg))

    residuals = y_pos - lr.predict(x2_pos)
    
    plt.hist(residuals, bins=20)
    plt.show()

    print('MSE', np.mean((y_neg - lr.predict(x2_neg))**2))

def main():
    task_4()
