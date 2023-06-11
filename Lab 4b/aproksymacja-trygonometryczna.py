import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

def func(x):
    return 10*3+(x**2)-10*3*np.cos(x)


def visualize(x, y, start, stop, n, m, function):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="data", color="red")
    X = np.arange(start, stop + 0.01, 0.01)
    plt.plot(X, func(X), label="Funkcja", color="red")
    plt.plot(X, function(X), label="Aproksymacja trygonometryczna", color="blue")
    plt.title(f"Aproksymacja trygonometryczna na {n} węzłach oraz m={m}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def max_error(Y1, Y2):
    return np.max([abs(Y1[i] - Y2[i]) for i in range(len(Y1))])

def sum_square_error(Y1, Y2):
    return np.sqrt(sum([(Y1[i] - Y2[i])**2 for i in range(len(Y1))]))/1000

class TrigonometricApproximation:
    def __init__(self, X, Y, n, m, start, stop):
        if m > np.floor((n - 1) / 2):
            raise Exception("m cannot be greater than floor of (n-1)/2")
        self.X = X
        self.Y = Y
        self.n = n
        self.m = m
        self.start = start
        self.stop = stop
        self.A = np.zeros(self.n)
        self.B = np.zeros(self.n)
        self.scale_to_2pi()
        self.compute_A_and_B()
        self.scale_from_2pi()

    def scale_to_2pi(self):
        range_length = self.stop - self.start
        for i in range(len(self.X)):
            self.X[i] /= range_length
            self.X[i] *= 2 * np.pi
            self.X[i] += -np.pi - (2 * np.pi * self.start / range_length)

    def compute_A_and_B(self):
        for i in range(self.n):
            ai = sum(self.Y[j] * np.cos(i * self.X[j]) for j in range(self.n))
            bi = sum(self.Y[j] * np.sin(i * self.X[j]) for j in range(self.n))
            self.A[i] = 2 * ai / self.n
            self.B[i] = 2 * bi / self.n

    def scale_from_2pi(self):
        range_length = self.stop - self.start
        for i in range(len(self.X)):
            self.X[i] -= -np.pi - (2 * np.pi * self.start / range_length)
            self.X[i] /= 2 * np.pi
            self.X[i] *= range_length

    def scale_point_to_2pi(self, x):
        range_length = self.stop - self.start
        x /= range_length
        x *= 2 * np.pi
        x += -np.pi - (2 * np.pi * self.start / range_length)
        return x

    def approximate(self, X):
        points = []
        for x in X:
            cp_x = deepcopy(x)
            cp_x = self.scale_point_to_2pi(cp_x)
            approximated_x = 1 / 2 * self.A[0] + sum(self.A[j] * np.cos(j * cp_x) + self.B[j] * np.sin(j * cp_x)
                                                     for j in range(1, self.m + 1))
            points.append(approximated_x)
        return points

def trig_approximation(start, stop, n, m):
    X = np.linspace(start, stop, n)
    Y = func(X)
    trigonometric_approximation = TrigonometricApproximation(X, Y, n, m, start, stop)
    visualize(X, Y, start, stop, n, m, trigonometric_approximation.approximate)

# trig_approximation(-4*np.pi, 4*np.pi+0.01, 10, 4)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 15, 5)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 15, 7)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 20, 9)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 25, 12)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 30, 8)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 30, 14)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 50, 24)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 80, 30)
# trig_approximation(-4*np.pi, 4*np.pi+0.01, 100, 37)

def calculate_error(start, stop, m):
    nodes = [5, 7, 10, 20, 30, 50, 80, 100]
    result = [None for _ in range(2 * len(m) * len(nodes))]

    total_X = np.linspace(start, stop, 1000)
    func_val = func(total_X)
    idx = 0

    for n in nodes:
        X = np.linspace(start, stop, n)
        Y = func(X)
        for i in m:
            if i > np.floor((n-1)/2):
                result[idx] = np.nan
                result[idx+1] = np.nan
            else:
                trigonometric_approximation = TrigonometricApproximation(X, Y, n, i, start, stop)
                trig_appr_result = trigonometric_approximation.approximate(total_X)
                result[idx] = max_error(trig_appr_result, func_val)
                result[idx + 1] = sum_square_error(trig_appr_result, func_val)
            idx += 2
    df = pd.DataFrame(data={"n": [val for val in nodes for _ in range(len(m))],
                            "m": m * len(nodes),
                            "trig approximation max error": result[::2],
                            "trig approximation sum square error": result[1::2]})
    return df

df = calculate_error(-4*np.pi, 4*np.pi+0.01, [2, 3, 5, 10, 14, 20, 24])
print(df)
print(df["trig approximation max error"].idxmin())
print(df["trig approximation sum square error"].idxmin())
df.to_csv("trig_approximation.csv", index=False)

trig_approximation(-4*np.pi, 4*np.pi+0.01, 100, 5)