import numpy as np

def mse(A, Y):
    return np.square(np.subtract(A,Y)).mean()

def crossentropy(A, Y):
    m = Y.shape[1]
    logp1 = np.multiply(np.log(A),Y)
    c1 = -1*np.sum(logp1)
    logp2 = np.multiply(np.log(1-A), (1-Y))
    c2 = -1*np.sum(logp2)
    return (c1+c2)/m   