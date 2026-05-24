import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def obtener_datos():
    # Carga y limpieza básica
    df = pd.read_csv('bmw.csv')
    df['model'] = df['model'].str.strip()
    df_filtrado = df[df['model'].isin(['3 Series', '1 Series'])].copy()

    # Transformación a Problema de Clasificación
    medianas = df_filtrado.groupby(['model', 'year'])['price'].median().reset_index()
    medianas.rename(columns={'price': 'precio_mediano'}, inplace=True)
    df_filtrado = pd.merge(df_filtrado, medianas, on=['model', 'year'], how='left')

    def categorizar_precio(row):
        if pd.isna(row['precio_mediano']) or row['precio_mediano'] == 0: return 1
        ratio = row['price'] / row['precio_mediano']
        if ratio < 0.85: return 0
        elif ratio <= 1.15: return 1
        else: return 2

    df_filtrado['target'] = df_filtrado.apply(categorizar_precio, axis=1)
    df_final = df_filtrado.drop(columns=['price', 'precio_mediano'])

    # Separar y Preprocesar
    X = df_final.drop('target', axis=1)
    y = df_final['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    cols_numericas = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    cols_categoricas = ['model', 'transmission', 'fuelType']

    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cols_numericas),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cols_categoricas) 
        ])

    X_train_procesado = preprocesador.fit_transform(X_train)
    X_test_procesado = preprocesador.transform(X_test)

    conteo = df_final['target'].value_counts().sort_index()
    total_coches = len(df_final)

    print("\n" + "=" * 50)
    print("DISTRIBUCIÓN TRAS CALCULAR LAS MEDIANAS:")
    print("=" * 50)
    print(f"Total de vehículos filtrados: {total_coches}\n")
    print(f"    Clase 0 (Ganga): {conteo[0]} muestras ({conteo[0]/total_coches*100:.2f}%)")
    print(f"    Clase 1 (Precio Justo): {conteo[1]} muestras ({conteo[1]/total_coches*100:.2f}%)")
    print(f"    Clase 2 (Caro): {conteo[2]} muestras ({conteo[2]/total_coches*100:.2f}%)")
    print("\n")

    # Devolvemos las 4 variables clave
    return X_train_procesado, X_test_procesado, y_train, y_test

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = obtener_datos()
    print("Datos procesados correctamente con forma:", X_tr.shape)