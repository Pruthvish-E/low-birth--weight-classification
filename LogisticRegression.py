import numpy as np
from scipy.optimize import fmin_tnc
def sigmoid(X):
    # activation function to map real value between 0 and 1
    return 1 / (1 + np.exp(-X))

def cost_function(theta, X, Y):
    # computes the cost function for instances
    m = X.shape[0]
    total_cost = -(1 / m) * np.sum(Y * np.log(sigmoid(np.dot(X, theta))) + (1 - Y) * np.log(1 - sigmoid(np.dot(X, theta))))
    return total_cost

def gradient(theta, X, Y):
    # compute gradient descent at point theta of cost function
    m = X.shape[0]
    return (1 / m) * np.dot(X.T, sigmoid(np.dot(X, theta)) - Y)

def fit(X, Y, theta):   
    # this function is used to find the model parameters that minimizes cost function
    # fmin_tnc is function from scipy which is used to compute the min for any function
    optimum_weight = fmin_tnc(func=cost_function, x0=theta, fprime=gradient,args=(X, Y.flatten()))
    return optimum_weight[0]

def predict(X):
    # computes the weighted sum of inputs mapped to value between 0 and 1
    theta = X[:, np.newaxis]
    return sigmoid(np.dot(X, theta))

def accuracy(X, Y, prob_threshold = 0.5):
    result = predict(X) >= prob_threshold 
    predicted_classes = (result.astype(int)).flatten()
    accuracy = np.mean(predicted_classes == Y)
    return 100 * accuracy