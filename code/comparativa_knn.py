import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Importamos la función limpia que creamos antes
from preprocesado_bmw import obtener_datos

def main():
    print("=" * 65)
    print("EJECUTANDO MODELO BASE DE COMPARACIÓN: K-NEAREST NEIGHBORS (K-NN)")
    print("=" * 65)

    # 1. Cargar exactamente los mismos datos preprocesados
    print("Importando y procesando datos desde el pipeline...")
    X_train_procesado, X_test_procesado, y_train, y_test = obtener_datos()

    # 2. Inicializar el clasificador K-NN
    # Usamos 5 vecinos (un estándar de la industria muy robusto)
    knn_clf = KNeighborsClassifier(n_neighbors=5)

    # 3. Entrenar el modelo (scikit-learn lo hace de forma instantánea)
    print("Entrenando K-NN...")
    knn_clf.fit(X_train_procesado, y_train)
    print("¡Entrenamiento completado!")

    # 4. Evaluar rendimiento
    pred_train_knn = knn_clf.predict(X_train_procesado)
    pred_test_knn = knn_clf.predict(X_test_procesado)

    acc_train_knn = accuracy_score(y_train, pred_train_knn)
    acc_test_knn = accuracy_score(y_test, pred_test_knn)


    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    print(f"Precisión K-NN en Entrenamiento (Train Accuracy): {acc_train_knn * 100:.2f}%")
    print(f"Precisión K-NN en Test (Test Accuracy): {acc_test_knn * 100:.2f}%")

    # =========================================
    # NUEVO: Gráfica "Pro" de la Matriz para K-NN
    # ==========================================
    etiquetas = ['Ganga', 'Justo', 'Caro']

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, 
        pred_test_knn, # Ojo: Aquí usamos la predicción del K-NN
        display_labels=etiquetas, 
        cmap=plt.cm.Greens # Podemos usar tonos verdes para diferenciarla de la otra
    )

    disp.ax_.set_title('Matriz de Confusión - K-Nearest Neighbors')

    plt.show()

if __name__ == "__main__":
    main()