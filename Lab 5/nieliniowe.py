import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.float_format = "{:,.12f}".format

def func1(x):
    n = 15
    m = 10
    y = x**n - (1-x)**m
    return y

def der_func1(x):
    n = 15
    m = 10
    deriv = n*x**(n-1) + m*(1-x)**(m-1)
    return deriv

plt.figure(figsize=(8,6))
X = np.arange(0.2, 2+0.01, 0.01)
plt.plot(X, func1(X), label = "Function")
plt.title(f"Plot of the function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()

def newtons_method(func, der, x_0, epsilon, max_iter, stop_condition):
    x_n = x_0
    for n in range(max_iter):
        f_xn = func(x_n)
        der_fxn = der(x_n)
        if der_fxn == 0:
            # Zero derivative. No solution found.
            return None, None
        if stop_condition == "abs" and abs(f_xn) < epsilon:
            # Found solution
            return x_n, n
        elif stop_condition == "points" and abs(f_xn / der_fxn) < epsilon:
            # Found solution
            return x_n, n
        x_n -= f_xn / der_fxn
    # Exceeded maximum number of iterations. No solution found.
    return None, np.inf
def secant(func, x_1, x_2, epsilon, max_iter, stop_condition):
    for n in range(max_iter):
        if func(x_1) == func(x_2):
            # Divided by zero
            return None, None
        x_1, x_2 = x_2, x_2 - (x_2 - x_1) * func(x_2) / (func(x_2) - func(x_1))
        if stop_condition == "abs" and abs(func(x_2)) < epsilon:
            # Found solution
            return x_2, n
        elif stop_condition == "points" and abs(x_1 - x_2) < epsilon:
            # Found solution
            return x_2, n
    # Exceeded maximum number of iterations. No solution found.
    return None, np.inf

def create_dataframe(method_name, epsilon, max_iter, stop_condition):
    X = np.arange(0.2, 2 + 0.1, 0.1)
    print(X)
    result = []
    for x_0 in X:
        if method_name == "newton":
            x, n = newtons_method(func1, der_func1, x_0, epsilon, max_iter, stop_condition)
            result += [x, n, x_0]
        elif method_name == "secant":
                x, n = secant(func1, x_0, 2, epsilon, max_iter, stop_condition)
                result += [x, n, x_0]
                x, n = secant(func1, 0.2, x_0, epsilon, max_iter, stop_condition)
                result += [x, n, x_0]
    df = pd.DataFrame(data={"x value": result[::3],
                            "num of iterations": result[1::3],
                            "point": result[2::3]})
    return df

# print(create_dataframe("newton", 10**(-15), 100, "points"))
# print(create_dataframe("newton", 10**(-4), 100, "points"))
# print(create_dataframe("newton", 10**(-15), 100, "abs"))
# print(create_dataframe("newton", 10**(-4), 100, "abs"))
print(create_dataframe("secant", 10**(-15), 100, "points"))