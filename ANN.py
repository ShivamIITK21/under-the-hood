import numpy as np
import pandas as pd
import activations as ac
import costs

class NeuralNetwork():

    def __init__(self, learning_rate:float, input_size:int, work:object, layerSize:int, params=0, output_size = 1):

        assert learning_rate > 0
        assert (work == 'regression' or work == 'classification')
        assert layerSize > 0

        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.work = work
        self.layerSize = layerSize
        self.params = params

    def initializeParameters(self):
        W1 = np.random.randn(self.layerSize, self.input_size)*0.01
        b1 = np.zeros((self.layerSize, 1))
        W2 = np.random.randn(self.output_size, self.layerSize)
        b2 = np.zeros((self.output_size, 1))
        
        return {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

    def forwardPropagation(self,params, X):
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        Z1 = np.matmul(W1, X) + b1
        A1 = ac.relu(Z1)
        Z2 = np.matmul(W2,A1) + b2
        A2 = Z2
        if self.work == 'regression':
            A2 = ac.identity(Z2)
        if self.work == 'classification':
          A2 = ac.sigmoid(Z2)

        return A2, {'Z1':Z1, 'A1':A1, 'Z2':Z2, 'A2':A2}

    def getCost(self, A2, Y):
        cost = 0
        if self.work == 'regression':
            cost = costs.mse(A2, Y)

        if self.work == 'classification':
            cost = costs.crossentropy(A2, Y)
 
        return cost

    def backwardPropagation(self, params, cache, X, Y):
        W2 = params['W2']

        Z1 = cache['Z1']
        A1 = cache['A1']
        A2 = cache['A2']

        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = (1/m)*(np.matmul(dZ2, A1.T))
        db2 = (1/m)*(np.sum(dZ2, axis = 1, keepdims = True))
        dZ1 = (np.matmul(W2.T, dZ2))*ac.derivativeRelu(Z1)
        dW1 = (1/m)*(np.matmul(dZ1, X.T))
        db1 = (1/m)*(np.sum(dZ1, axis = 1, keepdims = True))

        return {'dW1':dW1, 'db1':db1, 'dW2':dW2, 'db2':db2}

    def update(self, params, grads):
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        W1 -= self.learning_rate*grads['dW1']
        b1 -= self.learning_rate*grads['db1']
        W2 -= self.learning_rate*grads['dW2']
        b2 -= self.learning_rate*grads['db2']

        return {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

    def train(self, X, Y, epochs):
        params = self.initializeParameters()

        for i in range(0, epochs):
            A2, cache = self.forwardPropagation(params, X)
            cost = self.getCost(A2, Y)
            grads = self.backwardPropagation(params, cache, X, Y)
            params = self.update(params, grads)
            print("Cost = "+ str(cost) + " epoch = " + str(i))

        self.params = params

    def predict(self, X):
        pred = 0
        cache = 0
        if self.work == 'regression':
            pred, cache = self.forwardPropagation(self.params, X)
            return pred

        if self.work == 'classification':
            pred, cache = self.forwardPropagation(self.params, X)
            pred = (pred > 0.1).astype(int)

        return pred    
