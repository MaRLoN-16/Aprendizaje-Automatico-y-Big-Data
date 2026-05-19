import subprocess
import sys
import time

def ejecutar_fase(script_name, descripcion):
    print("\n" + "🌟" * 30)
    print(f"🚀 INICIANDO FASE: {descripcion}")
    print("🌟" * 30)
    time.sleep(1) # Pequeña pausa para que se lea bien en terminal
    
    try:
        # sys.executable asegura que usa el mismo entorno (aa-env) que estás usando tú
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ ERROR: Hubo un fallo al ejecutar {script_name}. Revisa el código de ese archivo.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo {script_name}. ¿Estás en la carpeta correcta?")
        sys.exit(1)

def main():
    print("==========================================================")
    print("🚗 PIPELINE DE MACHINE LEARNING: CLASIFICADOR PRECIOS BMW ")
    print("👨‍💻 Autor: Marlon Chirino Mijenes")
    print("==========================================================\n")
    
    print("Iniciando la ejecución completa del proyecto...")
    time.sleep(1)
    
    # Fase 1: Entrenamiento de la Red Neuronal (Este script ya importa el preprocesado por dentro)
    ejecutar_fase("entrenamiento_bmw.py", "Red Neuronal desde cero (NumPy)")
    
    # Fase 2: Ejecución del Modelo Base para comparar
    ejecutar_fase("comparativa_knn.py", "Línea base comparativa (K-NN con Scikit-learn)")
    
    print("\n" + "=" * 58)
    print("✅ EJECUCIÓN DEL PIPELINE COMPLETADA CON ÉXITO")
    print("📊 ¡Todos los datos listos para redactar las conclusiones!")
    print("==========================================================")

if __name__ == "__main__":
    main()