import numpy as np
import matplotlib.pyplot as plt
import utils
import linear_reg
import linear_reg_vectorized
import public_tests

public_tests.compute_cost_test(linear_reg.compute_cost)
public_tests.compute_gradient_test(linear_reg.compute_gradient)

public_tests.compute_cost_test(linear_reg_vectorized.compute_cost)
public_tests.compute_gradient_test(linear_reg_vectorized.compute_gradient)

try:
    x_train, y_train = utils.load_data()
except FileNotFoundError:
    print("No se ha encontrado 'data/ex1data1.txt'.")
    exit()

w_init = 0.0
b_init = 0.0
iterations = 1500
alpha = 0.01

print(f"\nEntrenando el modelo con alpha={alpha} por {iterations} iteraciones...")
w_final, b_final, J_hist = linear_reg.gradient_descent(x_train, y_train, w_init, b_init, linear_reg.compute_cost, linear_reg.compute_gradient, alpha, iterations)

print(f"\nParametros encontrados: w = {w_final:.4f}, b = {b_final:.4f}")

predict1 = 3.5 * w_final + b_final
print(f'\nPara una población de 35,000, obetendriamos un beneficio de ${predict1*10000:.2f}')

predict2 = 7.0 * w_final + b_final
print(f'Para una población de 70,000, obetendriamos un beneficio de ${predict2*10000:.2f}')

plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, marker='x', c='r')
plt.plot(x_train, w_final * x_train + b_final, c='b')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.legend()
plt.show()