import numpy as np
import matplotlib.pyplot as plt

# funkcja licząca zgodnie z metodą lagrange'a
def lagrange_interpolation(x, y, x_int):
    n = len(x)
    L = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i] *= (x_int - x[j])/(x[i] - x[j])
    return np.sum(y*L)

# funkcja licząca zgodnie z metodą newton'a
def newton_interpolation(x, y, x_int):
    n = len(x)
    f = np.zeros((n,n))
    f[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            f[i][j] = (f[i+1][j-1] - f[i][j-1])/(x[i+j] - x[i])
    p = f[0][n-1]
    for j in range(1,n):
        p = p*(x_int - x[n-1-j]) + f[0][n-1-j]
    return p

# główna funkcja uruchamiająca inne oraz rysująca wykres
def plot_function_and_interpolation(x, y,x_draw, y_draw, sampling, interpolation_method,node_method):
    x_int = np.linspace(x[0], x[-1], sampling) # punkty próbkowania
    y_int = np.zeros(sampling)
    error=0
    maxi=0
    for i in range(sampling):
        if interpolation_method == 'lagrange':
            y_int[i] = lagrange_interpolation(x, y, x_int[i])
        elif interpolation_method == 'newton':
            y_int[i] = newton_interpolation(x, y, x_int[i])
        error+=(y_draw[i]-y_int[i])**2
        maxi=max(maxi,np.abs(y_draw[i]-y_int[i]))
    error=np.sqrt(error)
    error/=500
    if interpolation_method == 'lagrange':
        plt.title('Lagrange z '+ str(len(x))+ " węzłami "+node_method)
    elif interpolation_method == 'newton':
        plt.title('Newton z '+ str(len(x))+ " węzłami "+node_method)
    plt.plot(x_draw, y_draw, label='Funkcja oryginalna')
    plt.plot(x_int, y_int, label='Funkcja interpolująca')
    plt.plot(x, y, 'o', label='Węzły')
    plt.plot([], [], ' ', label='Max różnica: '+str("%.4f" %maxi))
    plt.plot([], [], ' ', label='Odchylenie: ' + str("%.4f" %error))
    plt.legend()
    plt.show()


#parametry
sampling = 100
how_many_nodes=9
fr=np.pi*-4
to=np.pi*4

#wyznaczenie wartości w węzłach
x = np.linspace(fr, to, how_many_nodes)
y = np.zeros(len(x))
for i,val in enumerate(x):
    y[i]=10*3+(val**2)-10*3*np.cos(val)

#obliczanie rzeczywistej funkcji
x_draw = np.linspace(fr, to, sampling)
y_draw = np.zeros(len(x_draw))
for i,val in enumerate(x_draw):
    y_draw[i]=10*3+(val**2)-10*3*np.cos(val)

# wyznaczenie wartości w węzłach zgodnych z zerami wielomianu Czebyszewa
x_cheb = np.zeros(how_many_nodes)
for i in range(how_many_nodes):
    x_cheb[i]=((1/2)*(fr+to))+(1/2)*(to-fr)*np.cos((2*(i+1)-1)*np.pi/(2*how_many_nodes))
y_cheb = np.zeros(how_many_nodes)
for i,val in enumerate(x_cheb):
    y_cheb[i]=10*3+(val**2)-10*3*np.cos(val)

# wywołania dla równomiernie rozmieszczonych węzłów
plot_function_and_interpolation(x, y,x_draw, y_draw, sampling, 'lagrange','(równomiernie)')
plot_function_and_interpolation(x, y,x_draw, y_draw, sampling, 'newton','(równomiernie)')

# wywołania dla węzłów rozmieszczonych zgodnie z zerami wielomianu czebyszewa
plot_function_and_interpolation(x_cheb, y_cheb,x_draw, y_draw, sampling, 'lagrange','(Czebyszew)')
plot_function_and_interpolation(x_cheb, y_cheb,x_draw, y_draw, sampling, 'newton','(Czebyszew)')