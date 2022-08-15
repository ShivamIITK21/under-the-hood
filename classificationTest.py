import pandas as pd
import numpy as np
import ANN

data = pd.read_csv('classificationData.txt')
print(data)

X = data.iloc[:,:-1].values
Y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)

y_train = np.array([y_train])
y_test = np.array([y_test])

classifer = ANN.NeuralNetwork(0.005,X_train.T.shape[0],'classification', 16)
classifer.train(X_train,y_train, 10000)
print(classifer.predict(X_test))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test[0], classifer.predict(X_test)[0]))