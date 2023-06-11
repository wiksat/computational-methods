import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def func(x):
    return 10*3+(x**2)-10*3*np.cos(x)

plt.figure(figsize=(8,6))
X = np.arange(-4*np.pi, 4*np.pi+0.01, 0.01)
plt.plot(X, func(X), label = "Funkcja")
plt.title(f"Wykres funkcji")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
def visualize(x, y, start, stop, n, m, function):
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, label="Dane (węzły)", color="red")
    X = np.arange(start, stop+0.01, 0.01)
    plt.plot(X, func(X), label = "Funkcja",color="red")
    plt.plot(X, function(X), label = "Przybliżenie wielomianu metodą najmniejszych kwadratów", color="blue")
    plt.title(f"Przybliżenie wielomianu metodą najmniejszych kwadratów z {n} węzłami oraz m={m}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def max_error(Y1, Y2):
    return np.max([abs(Y1[i] - Y2[i]) for i in range(len(Y1))])

def sum_square_error(Y1, Y2):
    return np.sqrt(sum([(Y1[i] - Y2[i])**2 for i in range(len(Y1))]))/1000

def least_squares_approximation_func(X, Y, m):
    n = len(X)
    weights = [1] * n
    G = np.zeros((m, m))
    B = np.zeros(m)

    for j in range(m):
        for k in range(m):
            G[j, k] = sum(weights[i] * X[i] ** (j + k) for i in range(n))
        B[j] = sum(weights[i] * Y[i] * X[i] ** j for i in range(n))
    A = np.linalg.solve(G, B)
    return lambda x: sum(A[i] * x ** i for i in range(m))


def ls_approximation(start, stop, n, m):
    X = np.linspace(start, stop, n)
    Y = func(X)
    ls_appr_res = least_squares_approximation_func(X, Y, m+1)
    visualize(X, Y, start, stop, n, m, ls_appr_res)


ls_approximation(-4*np.pi, 4*np.pi+0.01, 5, 3)

ls_approximation(-4*np.pi, 4*np.pi+0.01, 12, 5)

ls_approximation(-4*np.pi, 4*np.pi+0.01, 15, 6)
ls_approximation(-4*np.pi, 4*np.pi+0.01, 15, 11)

ls_approximation(-4*np.pi, 4*np.pi+0.01, 25, 15)
ls_approximation(-4*np.pi, 4*np.pi+0.01, 40, 3)
ls_approximation(-4*np.pi, 4*np.pi+0.01, 45, 25)


# efekt Rungego
ls_approximation(-4*np.pi, 4*np.pi+0.01, 15,11)
ls_approximation(-4*np.pi, 4*np.pi+0.01, 15,20)


# najmniejszy błąd
ls_approximation(-4*np.pi, 4*np.pi+0.01, 150, 15)

def calculate_error(start, stop, m):
    nodes = [4, 5, 7, 10, 15, 20, 30, 50,75,100,150]
    result = [None for _ in range(2 * len(m) * len(nodes))]

    total_X = np.linspace(start, stop, 1000)
    func_val = func(total_X)
    idx = 0

    for n in nodes:
        X = np.linspace(start, stop, n)
        Y = func(X)
        for i in m:
            ls_approximation = least_squares_approximation_func(X, Y, i)
            ls_approximation_result = ls_approximation(total_X)
            result[idx] = max_error(ls_approximation_result, func_val)
            result[idx + 1] = sum_square_error(ls_approximation_result, func_val)
            idx += 2
    df = pd.DataFrame(data={"n": [val for val in nodes for _ in range(len(m))],
                            "m": m * len(nodes),
                            "ls approximation max error": result[::2],
                            "ls approximation sum square error": result[1::2]})
    return df

df = calculate_error(-4*np.pi, 4*np.pi+0.01, [2, 5, 8, 10, 12, 15])
# df.to_csv("plik.csv", index=False)
print(df)

# print(df["ls approximation max error"].idxmin())
# print(df["ls approximation sum square error"].idxmin())





