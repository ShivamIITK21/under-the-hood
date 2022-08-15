import pandas as pd
import numpy as np
import ANN

data = pd.read_csv('Power.csv')
print(data)

X = data.iloc[:,:-1].values
Y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train = np.array([y_train])
y_test = np.array([y_test])

print(X_train)
print(y_train)

regressor = ANN.NeuralNetwork(0.001,X_train.T.shape[0],'regression', 16)
regressor.train(X_train.T,y_train, 10000)
print(regressor.predict(X_test.T))

from sklearn.metrics import r2_score
print(r2_score(y_test[0], regressor.predict(X_test.T)[0]))