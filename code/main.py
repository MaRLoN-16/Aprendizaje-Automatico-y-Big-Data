import subprocess
import sys
import time
from entrenamiento_bmw import ejecutar_red_neuronal
from comparativa_knn import main as ejecutar_knn 

def ejecutar_fase(funcion_objetivo, descripcion):
    """
    Recibe la función a ejecutar y su descripción.
    """
    print("\n" + "🌟" * 30)
    print(f"🚀 INICIANDO FASE: {descripcion}")
    print("🌟" * 30)
    time.sleep(1) # Pequeña pausa para que se lea bien en terminal
    
    # Llamamos a la función directamente en memoria
    funcion_objetivo() 

def main():
    print("==========================================================")
    print("🚗 PIPELINE DE MACHINE LEARNING: CLASIFICADOR PRECIOS BMW ")
    print("👨‍💻 Autor: Marlon Chirino Mijenes")
    print("==========================================================\n")
    
    print("Iniciando la ejecución completa del proyecto...")
    time.sleep(1)
    
    # Fase 1: Entrenamiento de la Red Neuronal
    ejecutar_fase(ejecutar_red_neuronal, "Red Neuronal desde cero (NumPy)")
    
    # Fase 2: Ejecución del Modelo Base para comparar
    ejecutar_fase(ejecutar_knn, "Línea base comparativa (K-NN con Scikit-learn)")
    
    print("\n" + "=" * 58)
    print("✅ EJECUCIÓN DEL PIPELINE COMPLETADA CON ÉXITO")
    print("📊 ¡Todos los datos listos para redactar las conclusiones!")
    print("==========================================================")

if __name__ == "__main__":
    main()