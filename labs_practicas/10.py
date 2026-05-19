import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Cargamos el dataset 'epil' del paquete 'MASS' a través de get_rdataset
data = sm.datasets.get_rdataset("epil", "MASS").data

# En este dataset, la variable objetivo es 'y' y el grupo es 'subject'
y = data['y']
groups = data['subject']
X = data.drop(columns=['y', 'subject'])

X = pd.get_dummies(X, drop_first=True)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
X_test, y_test, groups_test = X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]

gkf = GroupKFold(n_splits=10)
for fold, (train_cv_idx, val_cv_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
    print(f"Fold {fold+1}: Train size={len(train_cv_idx)}, Val size={len(val_cv_idx)}")

models = {
    "MLP_1_capa": MLPRegressor(hidden_layer_sizes=(10,), max_iter=20000, random_state=42),
    "MLP_2_capas": MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=20000, random_state=42),
    "MLP_3_capas": MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=20000, random_state=42)
}

best_mse = float('inf')
best_model_name = ""
best_model = None

for name, model in models.items():
    mse_folds = []
    for train_cv_idx, val_cv_idx in gkf.split(X_train, y_train, groups_train):
        X_tr, y_tr = X_train.iloc[train_cv_idx], y_train.iloc[train_cv_idx]
        X_val, y_val = X_train.iloc[val_cv_idx], y_train.iloc[val_cv_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mse_folds.append(mean_squared_error(y_val, preds))
    
    mean_mse = np.mean(mse_folds)
    print(f"{name} - Mean CV MSE: {mean_mse:.4f}")
    
    if mean_mse < best_mse:
        best_mse = mean_mse
        best_model_name = name
        best_model = model

print(f"\nMejor modelo: {best_model_name}")
best_model.fit(X_train, y_train)
test_preds = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds)
print(f"Test MSE ({best_model_name}): {test_mse:.4f}\n")

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, groups=groups_train, cv=gkf, scoring='neg_mean_squared_error')
    print(f"{name} - cross_val_score MSE: {-scores.mean():.4f}")