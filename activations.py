import numpy as np

def relu(Z):
    return np.maximum(Z, 0)

def identity(Z):
    return Z

def sigmoid(Z):
    return 1/(1+np.exp(-Z))    

def derivativeRelu(Z):
    return np.greater(Z, 0).astype(int)        