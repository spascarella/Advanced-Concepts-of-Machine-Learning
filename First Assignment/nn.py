import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): # Sigmoid function for activation of neurons
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): # Derivative of sigmoid function for backpropagation
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, inputLayer, hiddenLayer, outputLayer, learningRate):
        self.inputLayer = inputLayer  # Number of neurons in input layer
        self.hiddenLayer = hiddenLayer  # Number of neurons in hidden layer
        self.outputLayer = outputLayer  # Number of neurons in output layer
        self.learningRate = learningRate

        self.weightsInputHidden = None
        self.weightsHiddenOutput = None

    def paramsInit(self):
        # Weights from input layer to hidden layer
        self.weightsInputHidden = np.random.rand(self.hiddenLayer, self.inputLayer)  

        # Weights from hidden to output layer
        self.weightsHiddenOutput = np.random.rand(self.outputLayer, self.hiddenLayer) 
        
        # Weights for biases nodes in Input and Hidden Layer
        self.biasHidden = np.random.rand(self.hiddenLayer, 1)  
        self.biasOutput = np.random.rand(self.outputLayer, 1)  

    def feedForward(self, X):
        Z1 = np.dot(self.weightsInputHidden, X) + self.biasHidden * 1 # Linear transformation of input layer
        A1 = sigmoid(Z1) # Activation function 
        Z2 = np.dot(self.weightsHiddenOutput, A1) + self.biasOutput  * 1 # Linear transformation of output layer
        prediction = sigmoid(Z2)

        return Z1, A1, Z2, prediction
    
    def backPropagation(self, X, Y, Z1, A1, Z2, prediction):
        outputLayerError = (prediction - Y) * sigmoid_derivative(Z2) 
        hiddenLayerError = (np.dot(self.weightsHiddenOutput.T, outputLayerError)) * sigmoid_derivative(Z1)

        # Gradient and update of the output layer weights
        # gradientOutputLayerWeight = np.dot(outputLayerError, A1.T)
        self.weightsHiddenOutput -= self.learningRate * np.dot(outputLayerError, A1.T)
        
        # Gradient and update of the hidden layer weights
        # gradientHiddenLayerWeight = np.dot(hiddenLayerError, X.T)
        self.weightsInputHidden -= self.learningRate * np.dot(hiddenLayerError, X.T)

        # Update biases by summing errors across all samples
        self.biasOutput -= self.learningRate * np.sum(outputLayerError, axis=1, keepdims=True)
        self.biasHidden -= self.learningRate * np.sum(hiddenLayerError, axis=1, keepdims=True)
    
    def train(self, X, Y, epochs):
        loss_history = []

        for epoch in range(epochs):
            Z1, A1, Z2, prediction = self.feedForward(X)
            self.backPropagation(X, Y, Z1, A1, Z2, prediction)

            loss = np.mean((Y - prediction) ** 2)
            loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return loss_history, prediction, Y


if __name__ == "__main__":
    nn = NeuralNetwork(8, 3, 8, 0.1)
    nn.paramsInit()
    X = np.eye(8)  # Identity matrix as input
    Y = X  # Target values
    Z1, A1, Z2, prediction = nn.feedForward(X)
    nn.backPropagation(X, Y, Z1, A1, Z2, prediction)

    epochs = 100000
    loss_history, prediction, y = nn.train(X, Y, epochs)

    print(prediction, y)
    plt.plot(range(epochs), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over time (epochs)')
    plt.show()
