from entrenamiento_bmw import ejecutar_red_neuronal
from comparativa_knn import main as ejecutar_knn 

def main():
    print("=" * 80)
    print("PIPELINE DE MACHINE LEARNING: CLASIFICADOR PRECIOS BMW ")
    print("Autor: Marlon Chirino Mijenes")
    print("=" * 80 + "\n")
    
    print("Iniciando la ejecución completa del proyecto...")
    
    # Entrenamiento de la Red Neuronal
    print("Iniciando Red Neuronal desde cero (NumPy):")
    ejecutar_red_neuronal() 
    
    # Ejecución del Modelo KNN para comparar
    print("Iniciando Comparativa (K-NN con Scikit-learn):")
    ejecutar_knn() 
    
    print("\n" + "=" * 50)
    print("Ejecución terminada")
    print("=" * 50)

if __name__ == "__main__":
    main()