import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# Ex 1) Load dataset and codify categorical variables
epilepsy = sm.datasets.get_rdataset("epil", "MASS").data
print(epilepsy)
# Separate features and target
y = epilepsy["y"]
X_raw = epilepsy.drop(columns=["subject","y"])
# One-hot encode categorical variables
X = pd.get_dummies(X_raw, drop_first=True)

# Ex 2) Split train-test by group
# We use the subject (patient ID) as grouping variable
subjects = epilepsy["subject"].to_numpy()  
# Get the list of unique patient IDs
unique_subject_ids = np.unique(subjects)
# Split the IDs (not the whole samples)
from sklearn.model_selection import train_test_split
train_group_ids, test_group_ids = train_test_split(
    unique_subject_ids, test_size=0.25, random_state=42)
# Build boolean masks to select samples
train_mask = np.isin(subjects, train_group_ids)
test_mask = np.isin(subjects, test_group_ids)
# Apply the boolean mask to project the IDs to samples 
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# Ex 3) Partition the training set in k=10 folds
from sklearn.model_selection import GroupKFold
gcv = GroupKFold(n_splits=10)
# Validation loop that print the train-val subsets
train_ids = subjects[train_mask]
for fold, (tr_idx,val_idx) in enumerate(gcv.split(X_train, y_train, train_ids)):
    print(f"Fold {fold+1}")
    print(f"Train samples idx: {tr_idx}")
    print(f"Validation samples idx: {val_idx}")

candidate_models = {
    "mlp_small": MLPRegressor(hidden_layer_sizes=(10,), random_state=42),
    "mlp_medium": MLPRegressor(hidden_layer_sizes=(10, 10), random_state=42),
    "mlp_deep": MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=42)
}

for nombre, modelo in candidate_models.items():
    mse_folds = []
    
    # Es necesario volver a instanciar o llamar a split para cada modelo
    for tr_idx, val_idx in gcv.split(X_train, y_train, train_ids):
        # Como X_train e y_train son DataFrames/Series de Pandas, usamos .iloc
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Entrenar el modelo con la fold de training
        modelo.fit(X_tr, y_tr)
        
        # Predecir sobre la fold de validación
        y_pred = modelo.predict(X_val)
        
        # Calcular y guardar el MSE
        mse = mean_squared_error(y_val, y_pred)
        mse_folds.append(mse)
        
    # Calcular el MSE medio de validación para el modelo actual
    mse_medio = np.mean(mse_folds)
    print(f"Modelo: {nombre} | MSE Medio de Validación: {mse_medio:.4f}")

    from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

results = {}
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
for name, mlp in candidate_models.items() :

    mse_scores = []
    for tx_idx, val_idx in gcv.split(X_train, y_train, groups = train_ids):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        model = make_pipeline(StandardScaler(), clone(mlp))
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, y_pred))

    results[name] = np.mean(mse_scores)
    print(f"Model: {name} mean MSE: {results[name]}")



best_mlp = candidate_models["mlp_medium"]
final_model = make_pipeline(StandardScaler(), best_mlp)
final_model.fit(X_train, y_train)
y_pred_test = final_model.predict(X_test)
test_mse = mean_squared_error(y_train, y_pred_test)

print(f"Best model MSE {test_mse}")