from sklearn.datasets import make_moons
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

from sklearn.tree import DecisionTreeClassifier
depth = 1
tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

feature_names = ["x1", "x2"]
importances = tree.feature_importances_
print("Importancia de las variables con depth=1")
for i,name in enumerate(feature_names):
    print(f"{name}: {importances[i]}")

def plot_classification(tree, depth, X, y, ax, n_points_mesh=300):
    # Define the region based on data samples
    x_lim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    y_lim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    
    # Create mesh points
    xx = np.linspace(*x_lim, n_points_mesh)
    yy = np.linspace(*y_lim, n_points_mesh)
    XX, YY = np.meshgrid(xx, yy)
    
    # Predict on grid
    Xtest = np.c_[XX.ravel(), YY.ravel()]
    Z = tree.predict(Xtest).reshape(XX.shape)
    
    # Plot de decision frontiers
    ax.contourf(XX, YY, Z, alpha=0.2)
    
    # Plot the samples
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=8, c='b', marker='v', alpha=0.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=8, c='r', marker='^', alpha=0.5)
    
    ax.set_title(f'depth={depth}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


import matplotlib.pyplot as plt

# Plot the decision frontiers on axis[0]
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

plot_classification(tree, depth, X, y, axes[0])

# Ex 4) Plot the decision tree on axis[1]
from sklearn.tree import plot_tree

plot_tree(tree, ax=axes[1], filled=True, feature_names=["x1", "x2"])

plt.show()