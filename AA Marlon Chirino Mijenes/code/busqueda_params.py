import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

from preprocesado_bmw import obtener_datos
from entrenamiento_bmw import randInitializeWeights, cost_opt, unroll_params, predict

def main():
    print("Obteniendo datos...")
    X_train, X_test, y_train, y_test = obtener_datos()
    
    input_layer_size = X_train.shape[1]
    num_labels = 3
    
    # Preparamos el One-Hot para el entrenamiento
    m_train = X_train.shape[0]
    y_train_onehot = np.zeros((m_train, num_labels))
    y_train_onehot[np.arange(m_train), y_train] = 1

    # Definimos los valores a probar
    lista_neuronas = [15, 25, 35]
    lista_lambdas = [0.01, 0.1, 1.0, 3.0]
    
    # Diccionario para guardar los resultados
    resultados = {n: {'train': [], 'test': []} for n in lista_neuronas}

    print("\n INICIANDO TEST DE HIPERPARÁMETROS...")
    print("Esto puede tardar varios minutos...\n")

    for neuronas in lista_neuronas:
        for lam in lista_lambdas:
            print(f"Entrenando -> Neuronas: {neuronas} | Lambda: {lam} ...", end=" ")
            
            # Inicializamos pesos
            i_Theta1 = randInitializeWeights(input_layer_size, neuronas)
            i_Theta2 = randInitializeWeights(neuronas, num_labels)
            i_params = np.concatenate([i_Theta1.ravel(), i_Theta2.ravel()])
            
            # Entrenamos con maxiter = 300
            res = minimize(fun=cost_opt, x0=i_params, 
                           args=(input_layer_size, neuronas, num_labels, X_train, y_train_onehot, lam), 
                           method='CG', jac=True, options={'maxiter': 300, 'disp': False})
            
            T1_fin, T2_fin = unroll_params(res.x, input_layer_size, neuronas, num_labels)
            
            # Predecir
            acc_train = accuracy_score(y_train, predict(T1_fin, T2_fin, X_train)) * 100
            acc_test = accuracy_score(y_test, predict(T1_fin, T2_fin, X_test)) * 100
            
            # Guardar
            resultados[neuronas]['train'].append(acc_train)
            resultados[neuronas]['test'].append(acc_test)
            
            print(f"Hecho! (Test Acc: {acc_test:.2f}%)")

    print("\nGenerando gráficas...")
    
    # Creamos 3 subgráficos en fila (1 fila, 3 columnas)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Búsqueda de Hiperparámetros: Precisión vs Lambda', fontsize=16, fontweight='bold', y=1.05)
    plt.style.use('ggplot') 

    for idx, neuronas in enumerate(lista_neuronas):
        ax = axes[idx]
        
        # Eje X serán las lambdas, Eje Y las precisiones
        ax.plot(lista_lambdas, resultados[neuronas]['train'], marker='o', color='royalblue', label='Train')
        ax.plot(lista_lambdas, resultados[neuronas]['test'], marker='s', color='darkorange', label='Validación (Test)')
        
        ax.set_title(f'{neuronas} Neuronas Ocultas', fontsize=12)
        ax.set_xlabel('Lambda (Regularización)')
        
        # Escala logarítmica para el eje X (porque lambda salta de 0.01 a 1.0 a 3.0)
        ax.set_xscale('log')
        
        if idx == 0:
            ax.set_ylabel('Accuracy (%)')
            ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()