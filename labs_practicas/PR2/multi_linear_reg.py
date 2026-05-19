import numpy as np
import copy
import math


def zscore_normalize_features(X):
  """
  computes  X, zcore normalized by column

  Args:
    X (ndarray (m,n))     : input data, m examples, n features

  Returns:
    X_norm (ndarray (m,n)): input normalized by column
    mu (ndarray (n,))     : mean of each feature
    sigma (ndarray (n,))  : standard deviation of each feature
  """
  mu = np.mean(X, axis=0)
  sigma = np.std(X, axis=0)
  X_norm = (X - mu) / sigma

  return (X_norm, mu, sigma)


def compute_cost(X, y, w, b):
  """
  compute cost
  Args:
    X (ndarray (m,n)): Data, m examples with n features
    y (ndarray (m,)) : target values
    w (ndarray (n,)) : model parameters  
    b (scalar)       : model parameter
  Returns
    cost (scalar)    : cost
  """
  """
  
  m = X.shape[0]
  cost = 0.0
  for i in range(m)
    f_wb_i = np.dot(X[i], w) + b
    cost = COST + (f_wb_i - y[i]) ** 2
  cost = cost/(2*m)
  """
  m = X.shape[0]
  
  # f_wb = X @ w + b
  f_wb = np.dot(X, w) + b

  cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)

  return cost


def compute_gradient(X, y, w, b):
  """
  Computes the gradient for linear regression 
  Args:
    X : (ndarray Shape (m,n)) matrix of examples 
    y : (ndarray Shape (m,))  target value of each example
    w : (ndarray Shape (n,))  parameters of the model      
    b : (scalar)              parameter of the model      
  Returns
    dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
    dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
  """
  m, n = X.shape

  predictions = np.dot(X, w) + b
  err = predictions - y

  dj_dw = (1 / m) * np.dot(X.T, err)

  dj_db = (1 / m) * np.sum(err)

  return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):
  """
  Performs batch gradient descent to learn theta. Updates theta by taking 
  num_iters gradient steps with learning rate alpha

  Args:
    X : (array_like Shape (m,n)    matrix of examples 
    y : (array_like Shape (m,))    target value of each example
    w_in : (array_like Shape (n,)) Initial values of parameters of the model
    b_in : (scalar)                Initial value of parameter of the model
    cost_function: function to compute cost
    gradient_function: function to compute the gradient
    alpha : (float) Learning rate
    num_iters : (int) number of iterations to run gradient descent
Returns
  w : (array_like Shape (n,)) Updated values of parameters of the model
      after running gradient descent
  b : (scalar)                Updated value of parameter of the model 
      after running gradient descent
  J_history : (ndarray): Shape (num_iters,) J at each iteration,
      primarily for graphing later
  """
  J_history = []
  w = copy.deepcopy(w_in)
  b = b_in
  
  for i in range(num_iters):
      dj_db, dj_dw = gradient_function(X, y, w, b)

      w = w - alpha * dj_dw
      b = b - alpha * dj_db
    
      if i < 100000:
          J_history.append(cost_function(X, y, w, b))

      if i % math.ceil(num_iters / 10) == 0:
          print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
  return w, b, J_history
