import math
import numpy as np
import matplotlib.pyplot as plt


def grad(x):
    return 2*x+ 5*np.cos(x)


def cost(x):
    return x**2 + 5*np.sin(x)


x_all = np.linspace(-10, 10, 100)


def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        new_plot(x_new, x[-1])
        x.append(x_new)
    return (x, it)


def new_plot(x_new, x_old):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x_all, cost(x_all), '-', label='Original data')
    plt.plot([x_new], [cost(x_new)], 'ro', label='Original data')
    plt.plot([x_old], [cost(x_old)], 'bo', label='Original data')
    plt.plot([x_new, x_old], [cost(x_new), cost(x_old)], 'r-', label='Original data')
    plt.legend()
    plt.show()


(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))

