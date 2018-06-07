import numpy as np
import matplotlib.pyplot as plt
from math import log

def relative_error(x0, x): return np.abs(x0 - x) / np.abs(x0)

x0 = np.logspace(-4, 4, 100, dtype = np.double)
eps = np.finfo(np.double).eps

def dec_to_log(x):
    return np.log(x)

def log_to_dec(x):
    return np.exp(x)

def power(x, y):
    return x * y

def sum(x, y):
    return dec_to_log(log_to_dec(x) + log_to_dec(y))

def sub(x, y):
    return dec_to_log(log_to_dec(x) - log_to_dec(y))

def mul(x, y):
    return x + y

def div(x, y):
    return x - y
    
def plot_error(x0, err):
    mask = np.logical_and(err > 0, err < np.inf)
    plt.loglog(x0[mask], err[mask], ".k")
    plt.loglog(x0, [eps] * len(err), "--r") # машинная точность для сравнения
    plt.xlabel("Значение аргумента")
    plt.ylabel("Относительная погрешность")
    plt.show()

def f_sqrt_sqr_log(x, n = 52):
    g = np.log(x)
    for k in range(n): 
        power(g, 2)
    for k in range(n):
        power(g, 0.5)
    return np.exp(g)


x = f_sqrt_sqr_log(x0)
err = relative_error(x0, x)
plot_error(x0, err)