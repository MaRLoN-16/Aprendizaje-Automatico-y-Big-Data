import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# EJERCICIO 1: Carga y partición del dataset
# ==========================================
print("Cargando dataset MNIST...")
mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target.astype('int')

# Split train-test (Código de la Imagen 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)


# ==========================================
# EJERCICIO 2: Funciones de Entrenamiento y Predicción
# ==========================================
# Train and return the k-NN model (Código de la Imagen 1)
def get_knn_fit_train(X_train, y_train, k=3, n_jobs=12):
    print(f"Traininig kNN model with k={k}")
    clf_model = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs)
    clf_model.fit(X_train, y_train)  # Train the model
    return clf_model


# Generate or load test predictions from cache (Código de la Imagen 2)
PRED_CACHE = Path('./preds_cache')
PRED_CACHE.mkdir(exist_ok=True)  # Create dir if it does not exist


def get_knn_predict_test(clf_model, X_test, k=3):
    pred_test_file = PRED_CACHE / f"y_pred_test_knn_{k}.npy"
    if pred_test_file.exists():
        print("Loading test predictions from", pred_test_file)
        y_pred_test = np.load(pred_test_file, allow_pickle=True)
    else:
        print(f"Generating test predictions for k={k}")
        y_pred_test = clf_model.predict(X_test)
        # CORRECCIÓN: Primero el archivo, luego los datos
        np.save(pred_test_file, y_pred_test)
    return y_pred_test


# ==========================================
# EJERCICIO 3: Evaluación del clasificador (k=3)
# ==========================================
clf_model_3 = get_knn_fit_train(X_train, y_train, k=3)
y_pred_3 = get_knn_predict_test(clf_model_3, X_test, k=3)

acc = accuracy_score(y_test, y_pred_3)
cm = confusion_matrix(y_test, y_pred_3)

print("\n--- Resultados para k=3 ---")
print(f"Accuracy global: {acc:.4f}")

# ==========================================
# EJERCICIO 4: Visualización de la matriz de confusión
# ==========================================
print("\nGenerando mapa de calor de la matriz de confusión...")
row_labels = [str(i) for i in range(10)]
col_labels = [str(i) for i in range(10)]

df_cm = pd.DataFrame(cm, index=row_labels, columns=col_labels)

plt.figure(figsize=(10, 8))
sb.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión (k=3)')
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.show()

# ==========================================
# EJERCICIO 5 y 6: Selección y gráfica del hiperparámetro k
# ==========================================
print("\nEvaluando rendimiento para k entre 1 y 10...")
valores_k = range(1, 11)
accuracies = []

for k_val in valores_k:
    modelo_k = get_knn_fit_train(X_train, y_train, k=k_val)
    pred_k = get_knn_predict_test(modelo_k, X_test, k=k_val)

    precision = accuracy_score(y_test, pred_k)
    accuracies.append(precision)
    print(f" -> Accuracy para k={k_val}: {precision:.4f}")

mejor_k = valores_k[np.argmax(accuracies)]
mejor_acc = max(accuracies)
print(f"\nEl valor que maximiza la accuracy es k={mejor_k} con {mejor_acc:.4f}")

# Representación gráfica del rendimiento
plt.figure(figsize=(8, 5))
plt.plot(valores_k, accuracies, marker='o', linestyle='-', color='r')
plt.title('Rendimiento de k-NN según k')
plt.xlabel('Número de vecinos (k)')
plt.ylabel('Accuracy en Test')
plt.xticks(valores_k)
plt.grid(True)
plt.show()