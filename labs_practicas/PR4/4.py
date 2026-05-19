import numpy as np
from scipy.io import loadmat
import multi_class
import utils

def main():
    try:
        data = loadmat('data/ex3data1.mat', squeeze_me=True)
        X = data['X']
        y = data['y']
    except FileNotFoundError:
        print("Error: 'data/ex3data1.mat' no encontrado.")
        return

    y[y == 10] = 0

    num_labels = 10
    lambda_ = 0.1
    
    print("Entrenando One-Vs-All...")
    all_theta = multi_class.oneVsAll(X, y, num_labels, lambda_)
    pred_ova = multi_class.predictOneVsAll(all_theta, X)
    
    accuracy_ova = np.mean(pred_ova == y) * 100
    print(f"Precisión One-Vs-All: {accuracy_ova:.2f}%")

    try:
        weights = loadmat('data/ex3weights.mat')
        Theta1 = weights['Theta1']
        Theta2 = weights['Theta2']
    except FileNotFoundError:
        print("Error: 'data/ex3weights.mat' no encontrado.")
        return

    print("Evaluando Red Neuronal...")
    pred_nn = multi_class.predict(Theta1, Theta2, X)
    accuracy_nn = np.mean(pred_nn == y) * 100
    print(f"Precisión Red Neuronal: {accuracy_nn:.2f}%")
    
    utils.displayData(X)

main()