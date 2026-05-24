import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def plot_decision_boundary(svm_clf, X, y, ax, degree=None):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, zorder=2)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svm_clf.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'], zorder=1)
    
    if isinstance(svm_clf, Pipeline):
        support_indices = svm_clf.named_steps['svc'].support_
        sv = X[support_indices]
    else:
        sv = svm_clf.support_vectors_
        
    ax.scatter(sv[:, 0], sv[:, 1], s=150, linewidth=1.5, facecolors='none', edgecolors='black', zorder=3)
    
    if degree is not None:
        ax.set_title(f"Polinómico Grado {degree}")

data = loadmat('data/ex6data1.mat')
X = data['X']
y = data['y'].ravel()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

clf1 = SVC(kernel='linear', C=1)
clf1.fit(X, y)
plot_decision_boundary(clf1, X, y, axes[0, 0])
axes[0, 0].set_title('Lineal, C = 1')

clf2 = SVC(kernel='linear', C=100)
clf2.fit(X, y)
plot_decision_boundary(clf2, X, y, axes[0, 1])
axes[0, 1].set_title('Lineal, C = 100')

clf3 = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("svc", SVC(kernel='linear', C=1))
])
clf3.fit(X, y)
plot_decision_boundary(clf3, X, y, axes[1, 0], degree=3)
axes[1, 0].set_title('Polinómica (Explícita) Grado 3, C = 1')

clf4 = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("svc", SVC(kernel='linear', C=100))
])
clf4.fit(X, y)
plot_decision_boundary(clf4, X, y, axes[1, 1], degree=3)
axes[1, 1].set_title('Polinómica (Explícita) Grado 3, C = 100')

plt.tight_layout()
plt.show()