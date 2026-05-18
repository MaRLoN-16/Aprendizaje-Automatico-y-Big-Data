import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def square(x):
    return x * x

def integrate_mc_iterative(fun, a, b, num_points=10000, plot = False):
    start_time = time.process_time()

    eje_x = np.linspace(a, b, num_points)
    eje_y = fun(eje_x)
    max_f = max(eje_y)
    min_f = min(eje_y)

    debajo = 0
    if plot: plt.figure()

    for i in range(num_points):
        x = np.random.uniform(a, b)
        y = np.random.uniform(min_f, max_f)
        if fun(x) > y:
            debajo = debajo + 1

    areatotal = abs(b - a) * abs(max_f - min_f)
    integral = (debajo / num_points) * areatotal

    end_time = time.process_time()
    elapsed_time = (end_time - start_time) * 1000

    return elapsed_time

def integrate_mc_vectorized(fun, a, b, num_points=10000, plot = False):
    start_time = time.process_time()

    eje_x = np.linspace(a, b, num_points)
    eje_y = fun(eje_x)
    max_f = max(eje_y)
    min_f = min(eje_y)

    if plot: plt.figure()
    
    random_x = np.random.uniform(a, b, num_points)
    random_y = np.random.uniform(min_f, max_f, num_points)
    f_random_x = fun(random_x)
    debajo = np.sum(random_y < f_random_x)
    area_total = abs(b-a) * abs(max_f - min_f)
    resultado = area_total * (debajo/num_points)
    end_time = time.process_time()

    elapsed_time = (end_time - start_time) * 1000

    return elapsed_time

def compara_tiempos():
    sizes = np.linspace(100, 10000, 20)
    sizes = sizes.astype('i')
    times_dot = []
    times_fast = []

    x1 = np.random.uniform(1, 100)
    x2 = np.random.uniform(1, 100)
    for size in sizes:
        times_dot.append(integrate_mc_iterative(square, x1, x2))
        times_fast.append(integrate_mc_vectorized(square, x1, x2))
    
    plt.figure()
    plt.scatter(sizes, times_dot, c='red', label='bucle')
    plt.scatter(sizes, times_fast, c='blue', label='vector')
    plt.legend()
    plt.xlabel('Tamaño del array')
    plt.ylabel('Tiempo (ms)')
    plt.show()

def main():
    a = 0.0
    b = 1.0
    N = 10000
    
    print(f"Calculando la integral de x*2 de {a} a {b} con N={N} puntos\n")

    time_loop = integrate_mc_iterative(square, a, b, N)
    print(f"Version iterativa:  (Tiempo: {time_loop} ms)")

    time_vec = integrate_mc_vectorized(square, a, b, N)
    print(f"Version vectorizada:  (Tiempo: {time_vec} ms)")

    res_scipy = integrate.quad(square, a, b)
    print(f"Resultado real: {res_scipy}")

    compara_tiempos()


main()