import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from src.tasks.task_1 import nearest
from src.tasks.task_2 import draw_plot
from src.tasks.task_3 import task_4

matplotlib.use('TkAgg')

def main():
    N = 1000
    K = 5
    x = [20, 20]

    df1 = 2 * pd.DataFrame(np.random.randn(N, 3), columns=['A', 'B','C'])
    df2 = 20 + 10 * pd.DataFrame(np.random.randn(N, 3), columns=['A', 'B','C'])

    data = np.concatenate((df1, df2), axis=0)
    data[:, 2] = 1
    data[0:N, 2] = 0

    s = 'b' * N + 'r' * N + 'g' * K + 'b'
    s_1 = np.full((N), '#FF0000')
    s_2 = np.full((N), '#00FF00')
    s_3 = np.full((K), '#0000FF')
    s_4 = ['#404000']
    s = np.concatenate((s_1, s_2, s_3, s_4), axis=0)
    
    # running tasks

    #k_best = nearest(x, K, data[:, 0:2])

    #draw_plot(K, data, 100)

    #print(k_best)
    #OX = np.concatenate((data[:, 0], k_best[:, 0], [x[0]]), axis=0) 
    #OY = np.concatenate((data[:, 1], k_best[:, 1], [x[1]]), axis=0)
    #plt.scatter(OX, OY, s=5, alpha=1, c=list(s))
    #plt.show()

    task_4()
