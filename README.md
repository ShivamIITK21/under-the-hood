# Implementation
1). This is the implementation of a one layered neural network with a single output neuron and variable sizes of input and the hidden layer.  
2). The model is capable of doing regression and binary-classification.  
3). All the necessary code is implemented in ANN.py, activations.py and costs.py, the other two files are the demonstrations of the model.  

# Approach
1). I have made a class called NeuralNetwork and the neural network is an instance of this class.  
2). All the neccessary functions like forward propagation, backward propagation, train and predict have been implemented in this class.  
3). I have made seperate files for my cost and activation functions.  

# How to use
1). Import ANN in your python file  
2). Create an instance of the NeuralNetwork class with the following attributes -> (learning rate, number of neurons in the input layer, the task(regression or classification), number of neurons in the hidden layer.  
3). Use the train method and specify the number of epochs, then use the predict method to predict a single or multiple values.  
4). I have created two test files to demonstrate the model.

NOTE - sklearn has only been used to verify the accuracy of the model and not in the actual implementation  

Task done by, Shivam Sharma, 210983