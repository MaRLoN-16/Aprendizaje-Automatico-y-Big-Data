import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import nn
import logistic_reg as lr
import utils

def unroll_params(nn_params, input_layer_size, hidden_layer_size, num_labels):
    theta1_end = hidden_layer_size * (input_layer_size + 1)
    
    Theta1 = np.reshape(nn_params[:theta1_end], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[theta1_end:], (num_labels, hidden_layer_size + 1))
    
    return Theta1, Theta2

def cost_opt(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    Theta1, Theta2 = unroll_params(nn_params, input_layer_size, hidden_layer_size, num_labels)
    
    J, grad1, grad2 = nn.backprop(Theta1, Theta2, X, y, lambda_)
    
    grad = np.concatenate([grad1.ravel(), grad2.ravel()])
    
    return J, grad

def main():
    input_layer_size  = 400
    hidden_layer_size = 25
    num_labels = 10
    
    try:
        data = loadmat('data/ex3data1.mat', squeeze_me=True)
        X = data['X']
        y_raw = data['y']
        y_raw[y_raw == 10] = 0
    except FileNotFoundError:
        print("ERROR: No se encuentra 'data/ex3data1.mat'.")
        return

    m = X.shape[0]
    
    y_onehot = np.zeros((m, num_labels))
    y_onehot[np.arange(m), y_raw] = 1

    try:
        weights = loadmat('data/ex3weights.mat')
        Theta1 = weights['Theta1']
        Theta2 = weights['Theta2']
        
        J = nn.cost(Theta1, Theta2, X, y_onehot, lambda_= 1)
        # J, grad1, grad2 = nn.backprop(Theta1, Theta2, X, y_raw, 1)
        utils.computeNumericalGradient(J, Theta1, Theta2)
        utils.checkNNGradients(nn.backprop)
        
    except FileNotFoundError:
        pass
    
    
main()