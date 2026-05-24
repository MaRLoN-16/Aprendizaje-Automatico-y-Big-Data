from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from preprocesado_bmw import obtener_datos
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    print("=" * 65)
    print("EJECUTANDO MODELO BASE DE COMPARACIÓN: K-NEAREST NEIGHBORS (K-NN)")
    print("=" * 65)

    # Cargar exactamente los mismos datos preprocesados
    print("Importando y procesando datos desde el pipeline...")
    X_train_procesado, X_test_procesado, y_train, y_test = obtener_datos()

    # Inicializar el clasificador K-NN, usamos 5 vecinos
    knn_clf = KNeighborsClassifier(n_neighbors=5)

    # Entrenar el modelo
    print("Entrenando K-NN...")
    knn_clf.fit(X_train_procesado, y_train)
    print("Entrenamiento completado!")
    print("\n")

    # Comparar rendimiento
    pred_train_knn = knn_clf.predict(X_train_procesado)
    pred_test_knn = knn_clf.predict(X_test_procesado)

    acc_train_knn = accuracy_score(y_train, pred_train_knn)
    acc_test_knn = accuracy_score(y_test, pred_test_knn)

    print(f"Precisión K-NN en Entrenamiento (Train Accuracy): {acc_train_knn * 100:.2f}%")
    print(f"Precisión K-NN en Test (Test Accuracy): {acc_test_knn * 100:.2f}%")

    print("\nReporte de Clasificación (Red Neuronal):")
    print(classification_report(y_test, pred_test_knn, target_names=['Ganga', 'Justo', 'Caro']))

    etiquetas = ['Ganga', 'Justo', 'Caro']

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, 
        pred_test_knn,
        display_labels=etiquetas, 
        cmap=plt.cm.Greens
    )

    disp.ax_.set_title('Matriz de Confusión - K-Nearest Neighbors')
    plt.show()

if __name__ == "__main__":
    main()