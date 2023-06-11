import matplotlib.pyplot as plt
import numpy as np


def all(n):
    def fun(val):
        return 10*3+(val**2)-10*3*np.cos(val)
    def dfun(val):
        return 2*val+30*np.sin(val)

    sampling = 100
    how_many_nodes=n
    fr=np.pi*-4
    to=np.pi*4

    #wyznaczenie wartości w węzłach
    x = np.linspace(fr, to, how_many_nodes)
    y = np.zeros(len(x))
    for i,val in enumerate(x):
        y[i]=fun(val)

    #obliczanie rzeczywistej funkcji
    x_draw = np.linspace(fr, to, sampling)
    y_draw = np.zeros(len(x_draw))
    for i,val in enumerate(x_draw):
        y_draw[i]=fun(val)
    # metoda hermite'a
    def hermie(nodes, f, df, x):
        n = len(nodes)
        z = []
        for i in range(n):
            z.append(nodes[i])
            z.append(nodes[i])
        n2 = 2*n
        matrix = np.zeros((n2, n2))
        for i in range(n2):
            for j in range(i+1):
                if j == 0:
                    matrix[i][j] = f(z[i])
                elif j == 1 & i % 2 == 1:
                    matrix[i][j] = df(z[i])
                else:
                    matrix[i][j] = matrix[i][j-1] - matrix[i-1][j-1]
                    matrix[i][j] = matrix[i][j] / (z[i] - z[i-j])

        result = 0
        helper = 1
        for i in range(n2):
            result = result + matrix[i][i] * helper
            helper = helper * (x - z[i])
        return result


    x_cheb = np.zeros(how_many_nodes)
    for i in range(how_many_nodes):
        x_cheb[i]=((1/2)*(fr+to))+(1/2)*(to-fr)*np.cos((2*(i+1)-1)*np.pi/(2*how_many_nodes))
    y_cheb = np.zeros(how_many_nodes)
    for i,val in enumerate(x_cheb):
        y_cheb[i]=fun(val)

    interpolated = []
    for i in range(sampling):
        interpolated.append(hermie(x, fun, dfun, x_draw[i]))
    maxi=0
    error=0

    for i in range(sampling):
        maxi=max(maxi,np.abs(y_draw[i]-interpolated[i]))
        error+=(y_draw[i]-interpolated[i])**2

    error=np.sqrt(error)
    error/=sampling

    plt.plot(x_draw, y_draw, label='Funkcja oryginalna')
    plt.title('Hermite z ' + str(len(x)) + " węzłami równoodległymi")
    plt.plot(x, y, 'o', label='Węzły')
    plt.plot(x_draw, interpolated, label='Funkcja interpolująca')
    plt.plot([], [], ' ', label='Max różnica: '+str("%.4f" %maxi))
    plt.plot([], [], ' ', label='Odchylenie: ' + str("%.4f" %error))
    plt.legend()
    plt.show()
    print('Max różnica: '+str("%.4f" %maxi))
    print('Odchylenie: ' + str("%.4f" %error))
# ilość węzłów
all(3)