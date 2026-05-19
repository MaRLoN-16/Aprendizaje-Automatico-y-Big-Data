import numpy as np
import logistic_reg as lr

def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """
    m = X.shape[0]
    
    a1 = np.hstack([np.ones((m, 1)), X])
    
    z2 = np.dot(a1, theta1.T)
    a2 = lr.sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2]) 
    
    z3 = np.dot(a2, theta2.T)
    a3 = lr.sigmoid(z3) 
    
    epsilon = 1e-15
    J_unreg = (-1.0 / m) * np.sum(y * np.log(a3 + epsilon) + (1.0 - y) * np.log(1.0 - a3 + epsilon))
    
    reg_term = (lambda_ / (2.0 * m)) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))
    
    J = J_unreg + reg_term

    return J

def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """
    m = X.shape[0]
    
    J = cost(theta1, theta2, X, y, lambda_)

    Delta1 = np.zeros_like(theta1)
    Delta2 = np.zeros_like(theta2)

    for k in range(m):
        a1 = np.insert(X[k], 0, 1).reshape(-1, 1) 
        
        z2 = np.dot(theta1, a1)
        a2 = np.insert(lr.sigmoid(z2), 0, 1).reshape(-1, 1)
        
        z3 = np.dot(theta2, a2)
        a3 = lr.sigmoid(z3)
        
        y_k = y[k].reshape(-1, 1)
        d3 = a3 - y_k
        
        d2 = np.dot(theta2[:, 1:].T, d3) * sigmoid_gradient(z2)
        
        Delta2 += np.dot(d3, a2.T)
        Delta1 += np.dot(d2, a1.T)

    grad1 = Delta1 / m
    grad2 = Delta2 / m
    
    grad1[:, 1:] += (lambda_ / m) * theta1[:, 1:]
    grad2[:, 1:] += (lambda_ / m) * theta2[:, 1:]

    return (J, grad1, grad2)

def sigmoid_gradient(z):
    g = lr.sigmoid(z)
    return g * (1.0 - g)