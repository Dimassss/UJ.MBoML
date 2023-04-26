import sympy as sym
from sympy.diffgeom import Manifold, Patch, CoordSystem, Differential
from sympy.diffgeom.rn import R2_r
from sympy.utilities.lambdify import lambdify
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import math

matplotlib.use('TkAgg')

class GDM:
    def __init__(self, f, n):
        # f: R^n -> R
        x = sym.symbols('x0:{}'.format(n))

        manifold = Manifold('M', n)
        patch = Patch('P', manifold)
        coords = CoordSystem('x', patch, x)

        df = Differential(f(coords.coord_functions()))

        self.df = lambda a: np.array([lambdify(x, df(e))(*a) for e in coords.base_vectors()])

    def next_step(self, a_n, alpha):
        return a_n - alpha * self.df(a_n)


def task_1():
    f = lambda x: x[0]**2
    gdm = GDM(f, n = 2)
    alpha = 0.2
    a = np.array([2])

    OX = np.linspace(-3,3, 100)
    OY = [f([x]) for x in OX]
    plt.plot(OX, OY)

    OX = [a[0]]

    for i in range(5):
        a = gdm.next_step(a, alpha)
        OX = OX + [a[0]]

    plt.plot(OX, [f([x]) for x in OX])

    plt.show()


def task_2_3():
    S = 50
    f = lambda x: x[0]**2 - 2*x[0] + x[1]**2
    a = [2,2]
    alpha = 0.2

    def create_surface(f, S):
        gdm = GDM(f, n = 2)

        # plot function in [-3,3]^2 * R space as (x, f(x)) for x in [-3,3]^2
        OX = np.linspace(-3,3, S)
        OY = np.linspace(-3,3, S)

        X, Y = np.meshgrid(OX, OY)
        Z = np.array([[f([X[i,j], Y[i,j]]) for j in range(S)] for i in range(S)])

        return [X, Y, Z]

    def create_convergence_line(f, a, alpha):
        gdm = GDM(f, n = 2)

        # using GDM to find local minimum and plot steps
        X = [a]
        Y = [f(a)]

        for j in range(5):
            a = gdm.next_step(a, alpha)
            X = X + [a]
            Y = Y + [f(a)]
        
        X = np.array(X)
        Y = np.array(Y)

        return [X[:, 0], X[:, 1], Y]

    def draw_3d(f, a, alpha, S):
        X, Y, Z = create_surface(f, S)
        Xp, Yp, Zp = create_convergence_line(f, a, alpha)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot_surface(X, Y, Z, color='red')
        ax.plot(Xp, Yp, Zp, color='blue', alpha=1)


    def draw_2d(f, a, alpha, S):
        X, Y, Z = create_surface(f, S)
        Xp, Yp, Zp = create_convergence_line(f, a, alpha)

        # create 2d plot
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.contour(X, Y, Z)
        ax.plot(Xp, Yp)


    draw_3d(f, a, alpha, S)
    plt.show()

    draw_2d(f, a, alpha, S)
    plt.show()


def task_4():
    def compute_error(a, b, x, y):
        return np.sum((y - (a*x + b))**2)

    import sklearn.linear_model as lm

    f = lambda x: (x**2)

    x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
    y = f(x) + np.random.randn(len(x))

    # aproximate linear function with GDM
    f = lambda v: compute_error(v[0], v[1], x, y)
    gdm = GDM(f, n=2)
    alpha = 0.02
    
    def aproximate_coefficients(start_point, nsteps):
        nonlocal gdm, alpha

        for i in range(nsteps):
            start_point = gdm.next_step(start_point, alpha)
        
        return start_point

    coefficients = [aproximate_coefficients(c, 100) for c in [[1,1], [0,0], [4,-4]]]
    #coefficients = [aproximate_coefficients([1, 1], 60)]

    # create Linear Regression model to compare it to our linear function which were gotten by GDM
    lr = lm.LinearRegression()
    lr.fit(x.reshape(-1,1), y.reshape(-1,1))

    # testing
    def plot_line_points(range_x, range_y, k, c):
        # y = kx + c
        # x = (y - c) / k

        y1, y2 = sorted(k * range_x + c)

        if y1 <= range_y[1] and y2 >= range_y[0]:
            y1 = min(range_y[1], max(range_y[0], y1))
            y2 = max(range_y[0], min(range_y[1], y2))

            X = (np.array([y1, y2]) - c) / k
            Y = k * X + c
            return [X, Y]
        else:
            return []

    range_x = np.array([min(x), max(x)])
    range_y = np.array([min(y), max(y)])

    fig = plt.figure()
    ax = fig.add_subplot()

    for c in coefficients:
        arr = plot_line_points(range_x, range_y, c[0], c[1])
        
        if len(arr) != 2:
            continue

        ax.plot(arr[0], arr[1])

    X, Y = plot_line_points(range_x, range_y, lr.coef_[0], lr.intercept_)
    ax.plot(X, Y)

    ax.scatter(x, y)

    plt.show()


    # plotting surface of squre error function to see why alpha must be so small.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    X, Y = np.meshgrid(np.linspace(-80, 80, 50), np.linspace(-80, 80, 50))
    Z = np.array([[f([X[i, j], Y[i, j]]) for i in range(50)] for j in range(50)])
    ax.plot_surface(X,Y,Z)
    plt.show()


def task_5():
    def compute_error(a, b, x, y):
        return np.sum((y - (a*x + b))**2)

    import sklearn.linear_model as lm

    f = lambda x: (x**2)

    x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
    y = f(x) + np.random.randn(len(x))

    # aproximate linear function with GDM
    f = lambda v: compute_error(v[0], v[1], x, y)
    gdm = GDM(f, n=2)
    alpha = 0.02

    from scipy import optimize
    import sklearn.linear_model as lm
    gdm = GDM(f, n=2)
    res = optimize.fmin_cg(f, np.array([0,0]), fprime=gdm.df)

    lr = lm.LinearRegression()
    lr.fit(x.reshape(-1,1), y.reshape(-1,1))
    b = np.array([lr.coef_[0][0], lr.intercept_[0]])

    print(res, b)
    print('l^2 distance between parameters as poits is', np.sum((b-res)**2)**0.5)


def main():
    task_5()