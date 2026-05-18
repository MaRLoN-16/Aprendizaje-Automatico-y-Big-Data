from sklearn.datasets import load_wine

wine = load_wine()
X = wine.data
y = wine.target

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA())
])

pipeline.fit(X)

pca = pipeline.named_steps['pca']
eigenvalues = pca.explained_variance_
iev = pca.explained_variance_ratio_
cev = np.cumsum(iev)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(1, len(iev) + 1), iev, alpha=0.5, label='IEV')
axes[0].plot(range(1, len(cev) + 1), cev, marker='o', label='CEV')
axes[0].legend()

axes[1].bar(range(1, len(eigenvalues) + 1), eigenvalues)
axes[1].axhline(y=1, color='g', linestyle='-')

plt.tight_layout()
plt.show()

print(f"Componentes para 50% varianza: {np.argmax(cev >= 0.50) + 1}")
print(f"Componentes para 75% varianza: {np.argmax(cev >= 0.75) + 1}")
print(f"Componentes para 95% varianza: {np.argmax(cev >= 0.95) + 1}")
print(f"Componentes criterio Kaiser: {np.sum(eigenvalues >= 1)}")