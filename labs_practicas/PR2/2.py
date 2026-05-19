import numpy as np
import matplotlib.pyplot as plt
from multi_linear_reg import compute_cost, compute_gradient, gradient_descent, zscore_normalize_features

data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
X_train = data[:, :4]
y_train = data[:, 4]

X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()