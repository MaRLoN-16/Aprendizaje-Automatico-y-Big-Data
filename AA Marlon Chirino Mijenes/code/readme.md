# Clasificador de Precios de Vehículos BMW (Machine Learning)

**Autor:** Marlon Chirino Mijenes

## Descripción

Este proyecto implementa un pipeline de Machine Learning para clasificar coches de la marca BMW de segunda mano según su precio en el mercado ("Ganga", "Precio Justo", "Caro"). El núcleo del proyecto es una Red Neuronal (MLP) implementada desde cero con `NumPy`, la cual es evaluada contra un modelo base K-Nearest Neighbors de `scikit-learn`.

## Requisitos y Entorno

Se recomienda utilizar un entorno virtual (como conda). Para instalar las librerías necesarias, se puede ejecutar:

`pip install -r requirements.txt`

## Dataset

Es necesario el dataset llamado "BMW Used Car Dataset" (https://www.kaggle.com/datasets/tahirmohd/bwm-car?select=bmw.csv), el archivo csv correspondiente ya se encuentra en la carpeta "code".

## Instrucciones de Ejecución (Punto de Entrada)
El punto de entrada principal para la ejecución del código es el fichero `main.py`. Este se encarga de ejecutar `preprocesado_bmw.py`, `entrenamiento_bmw.py` y `comparativa_knn.py`. Si se desea ejecutar el codigo que se utilizó para elegir los hiperpárametros, así como la cantidad de neuronas en la capa oculta, se puede ejecutar `busqueda_params.py` por separado. También se pueden ejecutar `preprocesado_bmw.py`, `entrenamiento_bmw.py` y `comparativa_knn.py` por separado.

## Estructura del Código

* `preprocesado_bmw.py`: Carga el dataset, genera las grupos mediante cálculo de medianas y aplica One-Hot Encoding y StandardScaler.

* `nn.py` y `logistic_reg.py`: Implementación matemática del Forward Propagation, Función de Coste y Backpropagation.

* `entrenamiento_bmw.py`: Configura y entrena la red neuronal usando `scipy.optimize`.

* `comparativa_knn.py`: Entrena el modelo K-NN para la comparativa.

* `busqueda_params.py`: Ejecuta la red neuronal con distintas configuraciones para encontrar la mejor opción.
