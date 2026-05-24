import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import nn 
import logistic_reg as lr
from preprocesado_bmw import obtener_datos

def randInitializeWeights(L_in, L_out):
    """Inicializa los pesos de una capa de forma aleatoria para romper la simetría"""
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

def unroll_params(nn_params, input_layer_size, hidden_layer_size, num_labels):
    """Desenrolla el vector gigante de Scipy en nuestras matrices Theta1 y Theta2"""
    theta1_end = hidden_layer_size * (input_layer_size + 1)
    Theta1 = np.reshape(nn_params[:theta1_end], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[theta1_end:], (num_labels, hidden_layer_size + 1))
    return Theta1, Theta2

def cost_opt(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    """Función envoltura para Scipy Optimize"""
    Theta1, Theta2 = unroll_params(nn_params, input_layer_size, hidden_layer_size, num_labels)
    # Llamamos a backprop de nn.py
    J, grad1, grad2 = nn.backprop(Theta1, Theta2, X, y, lambda_)
    # Scipy necesita que el gradiente sea un solo vector plano (1D)
    grad = np.concatenate([grad1.ravel(), grad2.ravel()])
    return J, grad

def predict(Theta1, Theta2, X):
    """Hace la pasada hacia adelante (Forward Prop) para predecir la clase"""
    m = X.shape[0]
    a1 = np.hstack([np.ones((m, 1)), X])
    z2 = np.dot(a1, Theta1.T)
    a2 = lr.sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])
    z3 = np.dot(a2, Theta2.T)
    a3 = lr.sigmoid(z3)
    # Devuelve el índice de la neurona con el valor más alto (0, 1 o 2)
    return np.argmax(a3, axis=1)

def ejecutar_red_neuronal():
    print("Importando y procesando datos...")
    # Llamamos a los datos DENTRO de la función para no dejar variables sueltas
    X_train_procesado, X_test_procesado, y_train, y_test = obtener_datos()

    # Parámetros de la arquitectura de la red
    input_layer_size = X_train_procesado.shape[1] # Columnas después del preprocesado
    hidden_layer_size = 35                        # Neuronas en la capa oculta
    num_labels = 3                                # Ganga (0), Justo (1), Caro (2)
    lambda_ = 0.1                                 # Parámetro de regularización

    print(f"\nIniciando Red Neuronal: {input_layer_size} Entradas -> {hidden_layer_size} Ocultas -> {num_labels} Salidas")

    # Convertir y_train (0, 1, 2) a One-Hot Encoding (matrices de 0s y 1s) para la red
    m_train = X_train_procesado.shape[0]
    y_train_onehot = np.zeros((m_train, num_labels))
    y_train_onehot[np.arange(m_train), y_train] = 1

    # Inicializar pesos aleatoriamente y enrollarlos para Scipy
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])

    print("Entrenando la red neuronal con Scipy Optimize (esto puede tardar unos minutos)...")
    opciones = {'maxiter': 1000, 'disp': True}
    
    resultados = minimize(
        fun=cost_opt, 
        x0=initial_nn_params, 
        args=(input_layer_size, hidden_layer_size, num_labels, X_train_procesado, y_train_onehot, lambda_), 
        method='CG', 
        jac=True, 
        options=opciones
    )

    # Extraer los pesos finales entrenados
    Theta1_final, Theta2_final = unroll_params(resultados.x, input_layer_size, hidden_layer_size, num_labels)
    print("\nEntrenamiento completado!")

    # Predecir en conjunto de entrenamiento
    pred_train = predict(Theta1_final, Theta2_final, X_train_procesado)
    acc_train = accuracy_score(y_train, pred_train)

    # Predecir en conjunto de test
    pred_test = predict(Theta1_final, Theta2_final, X_test_procesado)
    acc_test = accuracy_score(y_test, pred_test)

    print(f"\nPrecisión en Entrenamiento (Train Accuracy): {acc_train * 100:.2f}%")
    print(f"Precisión en Test (Test Accuracy): {acc_test * 100:.2f}%")
    

    print("\nReporte de Clasificación (Red Neuronal):")
    print(classification_report(y_test, pred_test, target_names=['Ganga', 'Justo', 'Caro']))

    etiquetas = ['Ganga', 'Justo', 'Caro']
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, 
        pred_test, 
        display_labels=etiquetas, 
        cmap=plt.cm.Blues
    )
    disp.ax_.set_title('Matriz de Confusión - Red Neuronal (NumPy)')

    plt.show()

if __name__ == "__main__":
    ejecutar_red_neuronal()