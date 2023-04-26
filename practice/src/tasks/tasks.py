import sympy as sym
from sympy.diffgeom import Manifold, Patch, CoordSystem, Differential
from sympy.diffgeom.rn import R2_r
from sympy.utilities.lambdify import lambdify
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

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


def task_2():
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


def main():
    task_2()