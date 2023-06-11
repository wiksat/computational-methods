import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from IPython.display import display

# dana funkcja
def func(x):
    return 10*3+(x**2)-10*3*np.cos(x)

# wizualizacja funkcji
# plt.figure(figsize=(8,6))
# X = np.arange(-4*np.pi, 4*np.pi+0.01, 0.01)
# plt.plot(X, func(X), label = "Funkcja")
# plt.title(f"Wykres funkcji")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# funkcja wizualizująca wszystkie wykresy
def visualize(x, y, start, stop, n, function, name):
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, label="Dane (węzły)", color="red")
    X = np.arange(start, stop+0.01, 0.01)
    plt.plot(X, func(X), label = "Funkcja oryginalna",color="red")
    if name == "own cubic spline":
        plt.plot(X, function(X), label = "Spline 3-go stopnia", color="blue")
        plt.title(f"Spline 3-go stopnia z {n} węzłami")
    elif name == "own quadratic spline":
        plt.plot(X, function(X), label = "Spline 2-go stopnia", color="blue")
        plt.title(f"Spline 2-go stopnia z {n} węzłami")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# funkcja licząca błąd maksymalny
def max_error(Y1, Y2):
    return max([abs(Y1[i] - Y2[i]) for i in range(len(Y1))])

# funkcja licząca błąd średniokwadratowy
def sum_square_error(Y1, Y2):
    return np.sqrt(sum([(Y1[i] - Y2[i])**2 for i in range(len(Y1))]))/1000

# klasa odpowiedzialna za licznie funkcji 3 stopnia
class CubicSpline:
    def __init__(self, X, Y, spline_type):
        self.X = X
        self.Y = Y
        self.n = len(X)
        self.sigma = None
        self.solve(spline_type)

    def h(self, i):
        return self.X[i + 1] - self.X[i]

    # różnicowe
    def delta(self, i):
        return (self.Y[i + 1] - self.Y[i]) / self.h(i)

    def delta2(self, i):
        return (self.delta(i + 1) - self.delta(i)) / (self.X[i + 1] - self.X[i - 1])

    def delta3(self, i):
        return (self.delta2(i + 1) - self.delta2(i)) / (self.X[i + 2] - self.X[i - 1])

    # uzupełnienie macierzu o warunki brzegowe
    def fill_boundaries(self, h_matrix, d_matrix, spline_type):
        if spline_type == "cubic":
            h_matrix[0][0] = -self.h(0)
            h_matrix[0][1] = self.h(0)
            h_matrix[self.n - 1][self.n - 2] = self.h(self.n - 2)
            h_matrix[self.n - 1][self.n - 1] = -self.h(self.n - 2)

            d_matrix[0] = np.power(self.h(0), 2) * self.delta3(0)
            d_matrix[self.n - 1] = -np.power(self.h(self.n - 2), 2) * self.delta3(self.n - 4)
            self.sigma = np.linalg.solve(h_matrix, d_matrix)
        elif spline_type == "natural":
            h_matrix = h_matrix[1:-1, 1:-1]
            d_matrix = d_matrix[1:-1]
            self.sigma = [0, *np.linalg.solve(h_matrix, d_matrix), 0]

    # przygotowanie macierzy
    def solve(self, spline_type):
        h_matrix = np.zeros(shape=(self.n, self.n))
        d_matrix = np.zeros(shape=(self.n, 1))
        for i in range(1, self.n - 1):
            h_matrix[i][i - 1] = self.h(i - 1)
            h_matrix[i][i] = 2 * (self.h(i - 1) + self.h(i))
            h_matrix[i][i + 1] = self.h(i)

            d_matrix[i] = self.delta(i) - self.delta(i - 1)

        self.fill_boundaries(h_matrix, d_matrix, spline_type)

    def find_interval(self, x):
        l = 0
        r = self.n - 1
        while l <= r:
            mid = (l + r) // 2
            if x >= self.X[mid]:
                l = mid + 1
            else:
                r = mid - 1
        return l - 1

    # do wizualizacji wyliczenie wsp
    def s(self, x):
        i = min(self.find_interval(x), self.n - 2)
        b = (self.Y[i + 1] - self.Y[i]) / self.h(i) - self.h(i) * (self.sigma[i + 1] + 2 * self.sigma[i])
        c = 3 * self.sigma[i]
        d = (self.sigma[i + 1] - self.sigma[i]) / self.h(i)
        return self.Y[i] + b * (x - self.X[i]) + c * np.power(x - self.X[i], 2) + d * np.power(x - self.X[i], 3)

    def S(self, xs):
        return [self.s(x) for x in xs]

# funkcja wywołująca klasę do liczenia spline'u 3-go stopnia
def own_cubic_spline(start, stop, n, spline_type):
    X = np.linspace(start, stop, n)
    Y = func(X)
    cubic_spline = CubicSpline(X, Y, spline_type)
    visualize(X, Y, start, stop, n, cubic_spline.S, "own cubic spline")

# Wywołanie funkcji 3-go stopnia (natural spline)
own_cubic_spline(-4*np.pi, 4*np.pi, 4, "natural")
own_cubic_spline(-4*np.pi, 4*np.pi, 10, "natural")
own_cubic_spline(-4*np.pi, 4*np.pi, 30, "natural")

# Wywołanie funkcji 3-go stopnia (cubic spline)
own_cubic_spline(-4*np.pi, 4*np.pi,4, "cubic")
own_cubic_spline(-4*np.pi, 4*np.pi,10, "cubic")
own_cubic_spline(-4*np.pi, 4*np.pi,30, "cubic")

# klasa odpowiedzialna za licznie funkcji 2 stopnia
class QuadraticSpline:
    def __init__(self, X, Y, spline_type):
        self.X = X
        self.Y = Y
        self.n = len(X)
        self.a = None
        self.b = None
        self.solve(spline_type)

    def gamma(self, i):
        return (self.Y[i] - self.Y[i - 1]) / (self.X[i] - self.X[i - 1])

    # natural spline
    def a_natural(self, i):
        return (self.b_natural(i + 1) - self.b_natural(i)) / (2 * (self.X[i + 1] - self.X[i]))

    def b_natural(self, i):
        if i == 0:
            return 0
        return 2 * self.gamma(i) - self.b_natural(i - 1)

    # clamped spline
    def a_clamped(self, i):
        return (self.b_clamped(i + 1) - self.b_clamped(i)) / (2 * (self.X[i + 1] - self.X[i]))

    def b_clamped(self, i):
        if i == 0:
            # return self.gamma(1)
            return (self.Y[1] - self.Y[0]) / (self.X[1] - self.X[0])
        return 2 * self.gamma(i) - self.b_clamped(i - 1)

    def solve(self, spline_type):
        if spline_type == "clamped":
            self.a = self.a_clamped
            self.b = self.b_clamped
        elif spline_type == "natural":
            self.a = self.a_natural
            self.b = self.b_natural

    def find_interval(self, x):
        l = 0
        r = self.n - 1
        while l <= r:
            mid = (l + r) // 2
            if x >= self.X[mid]:
                l = mid + 1
            else:
                r = mid - 1
        return l - 1

    def s(self, x):
        i = min(self.find_interval(x), self.n - 2)
        a = self.a(i)
        b = self.b(i)
        return a * np.power(x - self.X[i], 2) + b * (x - self.X[i]) + self.Y[i]

    def S(self, xs):
        return [self.s(x) for x in xs]


def own_quadratic_spline(start, stop, n, spline_type):
    X = np.linspace(start, stop, n)
    Y = func(X)
    quadratic_spline = QuadraticSpline(X, Y, spline_type)
    visualize(X, Y, start, stop, n, quadratic_spline.S, "own quadratic spline")

# Wywołanie funkcji 2-go stopnia (natural spline)

own_quadratic_spline(-4*np.pi, 4*np.pi, 4, "natural")
own_quadratic_spline(-4*np.pi, 4*np.pi, 10, "natural")
own_quadratic_spline(-4*np.pi, 4*np.pi, 30, "natural")

# Wywołanie funkcji 2-go stopnia (clamped spline)

own_quadratic_spline(-4*np.pi, 4*np.pi, 4, "clamped")
own_quadratic_spline(-4*np.pi, 4*np.pi, 10, "clamped")
own_quadratic_spline(-4*np.pi, 4*np.pi, 30, "clamped")


# funkcja licząca błędy
def calculate_error(start, stop):
    nodes = [4, 5, 7, 10, 15, 20, 30, 50, 70, 80, 100, 150, 200, 500]
    result = [None for _ in range(8 * len(nodes))]

    df_result = pd.DataFrame()
    total_X = np.linspace(start, stop, 1000)
    func_val = func(total_X)
    idx = 0
    for n in nodes:
        X = np.linspace(start, stop, n)
        Y = func(X)
        nat_cubic_spline = CubicSpline(X, Y, "natural")
        nat_cubic_result = [item for sublist in nat_cubic_spline.S(total_X) for item in sublist]
        result[idx] = max_error(nat_cubic_result, func_val)
        result[idx + 1] = sum_square_error(nat_cubic_result, func_val)

        cub_cubic_spline = CubicSpline(X, Y, "cubic")
        cub_cubic_result = [item for sublist in cub_cubic_spline.S(total_X) for item in sublist]
        result[idx + 2] = max_error(cub_cubic_result, func_val)
        result[idx + 3] = sum_square_error(cub_cubic_result, func_val)

        nat_quadr_spline = QuadraticSpline(X, Y, "natural")
        result[idx + 4] = max_error(nat_quadr_spline.S(total_X), func_val)
        result[idx + 5] = sum_square_error(nat_quadr_spline.S(total_X), func_val)

        cla_quadr_spline = QuadraticSpline(X, Y, "clamped")
        result[idx + 6] = max_error(cla_quadr_spline.S(total_X), func_val)
        result[idx + 7] = sum_square_error(cla_quadr_spline.S(total_X), func_val)
        idx += 8
    df = pd.DataFrame(data={"n": nodes,
                            "cubic natural max error": result[::8],
                            "cubic natural sum square error": result[1::8],
                            "cubic cubic max error": result[2::8],
                            "cubic cubic sum square error": result[3::8],
                            "quadratic natural max error": result[4::8],
                            "quadratic natural sum square error": result[5::8],
                            "quadratic clamped max error": result[6::8],
                            "quadratic clamped sum square error": result[7::8]})
    return df

# df = calculate_error(-4*np.pi, 4*np.pi)
# df.to_csv("plik.csv", index=False)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     display(df)


# wywołanie najlepszej funkcji
# own_cubic_spline(-4*np.pi, 4*np.pi, 500, "cubic")