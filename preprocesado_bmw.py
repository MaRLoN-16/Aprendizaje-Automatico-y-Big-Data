import pandas as pd
import numpy as np

# 1. Carga y limpieza básica
print("Cargando el dataset...")
df = pd.read_csv('bmw.csv')

# Quitar espacios en blanco molestos al principio del nombre de los modelos
df['model'] = df['model'].str.strip()

# Filtrar para no pasarnos de tamaño (Serie 3 y Serie 1)
df_filtrado = df[df['model'].isin(['3 Series', '1 Series'])].copy()
print(f"Total de coches tras el filtrado: {len(df_filtrado)}")

# 2. Transformación a Problema de Clasificación
# Calculamos el precio mediano agrupando por el modelo y el año de matriculación
medianas = df_filtrado.groupby(['model', 'year'])['price'].median().reset_index()
medianas.rename(columns={'price': 'precio_mediano'}, inplace=True)

# Unimos esa información a nuestro dataset de trabajo
df_filtrado = pd.merge(df_filtrado, medianas, on=['model', 'year'], how='left')

# Función para categorizar el precio (0: Ganga, 1: Justo, 2: Caro)
def categorizar_precio(row):
    # Si por algún motivo no hay mediana (ej. solo hay 1 coche de ese año), lo ponemos como Justo
    if pd.isna(row['precio_mediano']) or row['precio_mediano'] == 0:
        return 1
        
    ratio = row['price'] / row['precio_mediano']
    
    if ratio < 0.85:       # Menos del 85% de la media del mercado
        return 0 # Ganga
    elif ratio <= 1.15:    # Entre 85% y 115% de la media
        return 1 # Precio Justo
    else:                  # Más del 115% de la media
        return 2 # Caro

df_filtrado['target'] = df_filtrado.apply(categorizar_precio, axis=1)

# ¡PASO CRUCIAL! Borramos el precio real para no hacer "trampa" en el entrenamiento (data leakage)
df_final = df_filtrado.drop(columns=['price', 'precio_mediano'])

# Comprobamos cómo han quedado repartidas nuestras categorías
print("\nDistribución de las categorías (0=Ganga, 1=Justo, 2=Caro):")
print(df_final['target'].value_counts())
print("\nPrimeras filas del dataset listo:")
print(df_final.head())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Separar características (X) y objetivo (y)
X = df_final.drop('target', axis=1)
y = df_final['target'].values

# 2. Dividir en Entrenamiento (80%) y Test (20%)
# MUY IMPORTANTE: Usamos stratify=y para que haya el mismo % de gangas en Train y en Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Identificar qué columnas son de texto y cuáles son números
cols_numericas = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
cols_categoricas = ['model', 'transmission', 'fuelType']

# 4. Crear el motor de preprocesado
preprocesador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cols_numericas),
        # drop='first' evita crear columnas redundantes (si no es Automático ni Semiautomático, por descarte es Manual)
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cols_categoricas) 
    ])

# 5. Aplicar la transformación matemática
# OJO: Hacemos fit_transform() en Train para que aprenda las escalas...
X_train_procesado = preprocesador.fit_transform(X_train)

# ... pero solo hacemos transform() en Test para no hacer trampas y espiar los datos del futuro
X_test_procesado = preprocesador.transform(X_test)

print(f"\nForma de X_train ANTES de procesar: {X_train.shape}")
print(f"Forma de X_train DESPUÉS de procesar: {X_train_procesado.shape}")
print("¡Datos 100% listos para las Redes Neuronales!")