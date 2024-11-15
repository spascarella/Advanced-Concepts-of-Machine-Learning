import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    '''
    Sigmoid function for activation of neurons
    '''
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): 
    '''
    Derivative of sigmoid function for backpropagation
    '''
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, inputLayer, hiddenLayer, outputLayer):
        self.inputLayer = inputLayer  # Number of neurons in input layer
        self.hiddenLayer = hiddenLayer  # Number of neurons in hidden layer
        self.outputLayer = outputLayer  # Number of neurons in output layer

        self.weightsInputHidden = None  # Weights from input layer to hidden layer
        self.weightsHiddenOutput = None # Weights from hidden to output layer
        self.biasHidden = None # Bias from input to hidden layer
        self.biasOutput = None # Bias from hidden to output layer

    def paramsInit(self, method = "uniform", bias_init = 'random'):
        '''
        Parameters initialization
        '''
        if method == "uniform": # Uniform random initialization (default)
            self.weightsInputHidden = np.random.rand(self.hiddenLayer, self.inputLayer)
            self.weightsHiddenOutput = np.random.rand(self.outputLayer, self.hiddenLayer)
        elif method == "zero":  # Zero initialization
            self.weightsInputHidden = np.zeros((self.hiddenLayer, self.inputLayer))
            self.weightsHiddenOutput = np.zeros((self.outputLayer, self.hiddenLayer))
       
        # Weights initialization for biases nodes in Input and Hidden Layer
        if bias_init == "random": # Uniform random initialization (default)
            self.biasHidden = np.random.rand(self.hiddenLayer, 1) 
            self.biasOutput = np.random.rand(self.outputLayer, 1) 
        elif bias_init == "zero": # Zero bias initialization 
            self.biasHidden = np.zeros((self.hiddenLayer, 1))
            self.biasOutput = np.zeros((self.outputLayer, 1))
        elif bias_init == "high": # High bias initialization 
            self.biasHidden = np.ones((self.hiddenLayer, 1)) * 5  # High bias value (e.g., 5)
            self.biasOutput = np.ones((self.outputLayer, 1)) * 5  # High bias value (e.g., 5)

    def feedForward(self, X):
        '''
        Feed Forward phase
        '''
        Z1 = np.dot(self.weightsInputHidden, X) + self.biasHidden * 1 # Linear transformation of input layer
        A1 = sigmoid(Z1) # Activation function 
        Z2 = np.dot(self.weightsHiddenOutput, A1) + self.biasOutput  * 1 # Linear transformation of output layer
        prediction = sigmoid(Z2)

        return Z1, A1, Z2, prediction
    
    def backPropagation(self, X, Y, Z1, A1, Z2, prediction, learningRate):
        '''
        Backpropagation phase
        '''
        outputLayerError = (prediction - Y) * sigmoid_derivative(Z2) 
        hiddenLayerError = (np.dot(self.weightsHiddenOutput.T, outputLayerError)) * sigmoid_derivative(Z1)

        # Gradient and update of the output layer weights
        gradientOutputLayerWeight = np.dot(outputLayerError, A1.T)
        self.weightsHiddenOutput -= learningRate *gradientOutputLayerWeight
        
        # Gradient and update of the hidden layer weights
        gradientHiddenLayerWeight = np.dot(hiddenLayerError, X.T)
        self.weightsInputHidden -= learningRate * gradientHiddenLayerWeight

        # Update biases by summing errors across all samples
        self.biasOutput -= learningRate * np.sum(outputLayerError, axis=1, keepdims=True) 
        self.biasHidden -= learningRate * np.sum(hiddenLayerError, axis=1, keepdims=True) 
    
    def train(self, X, Y, epochs, learningRate):
        '''
        Training function
        '''
        loss_history = []

        for epoch in range(epochs):
            Z1, A1, Z2, prediction = self.feedForward(X)
            self.backPropagation(X, Y, Z1, A1, Z2, prediction, learningRate)

            loss = np.mean((Y - prediction) ** 2)
            loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return loss_history, prediction, Y

def learning_rate_experiment(nn, X, Y, epochs,learning_rates):
    '''
    Function used to experiment different learning rates values
    '''
    results = {}

    plt.figure(figsize=(10, 6))

    for lr in learning_rates:
        nn.paramsInit()  # Reinitialize parameters for each learning rate
        loss_history, prediction, Y = nn.train(X, Y, epochs, lr)
        results[lr] = loss_history  # Store the loss history for plotting

        # Plot the loss curve for the current learning rate
        plt.plot(loss_history, label=f'Learning Rate {lr}')

    # Display all loss curves on the same plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Time for Different Learning Rates')
    plt.legend()
    plt.show()

    # Print final predictions and target for analysis
    np.set_printoptions(precision=2, suppress=True)
    print("Final Predictions (for the last learning rate tested):\n", prediction)
    print("Target:\n", Y)

def weight_initialization_experiment(nn, X, Y, init_methods, epochs, learning_rate):
    '''
    Function used to experiment different weight values
    '''
    results = {}

    plt.figure(figsize=(10, 6))
    
    for method in init_methods:
        nn.paramsInit(method=method)  # Initialize parameters with the specified method
        loss_history, prediction, Y = nn.train(X, Y, epochs, learning_rate)
        results[method] = loss_history  # Store the loss history for analysis and plotting

        # Plot the loss curve for the current initialization method
        plt.plot(loss_history, label=f'Initialization: {method}')

    # Display all loss curves on the same plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Time for Different Weight Initializations')
    plt.legend()
    plt.show()

    # Print final predictions and target for analysis
    np.set_printoptions(precision=2, suppress=True)
    print("Final Predictions (for the last initialization method tested):\n", prediction)
    print("Target:\n", Y)

    return results

def bias_initialization_experiment(nn, X, Y, bias_initializations, epochs, learning_rate):
    '''
    Function used to experiment different bias values
    '''

    results = {}

    plt.figure(figsize=(10, 6))
    
    for bias_init in bias_initializations:
        nn.paramsInit(bias_init=bias_init)  # Initialize biases according to the specified method
        loss_history, prediction, Y = nn.train(X, Y, epochs, learning_rate)
        results[bias_init] = loss_history  # Store the loss history for analysis and plotting

        # Plot the loss curve for the current bias initialization method
        plt.plot(loss_history, label=f'Bias Initialization: {bias_init}')

    # Display all loss curves on the same plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Time for Different Bias Initializations')
    plt.legend()
    plt.show()

    # Print final predictions and target for analysis
    np.set_printoptions(precision=2, suppress=True)
    print("Final Predictions (for the last bias initialization tested):\n", prediction)
    print("Target:\n", Y)

    return results

def noise_experiment(nn, X, Y, noise_levels, epochs, learning_rate):
    '''
    Function used to experiment noise in the input data
    '''

    results = {}

    plt.figure(figsize=(10, 6))
    
    for noise_level in noise_levels:
        nn.paramsInit()  # Reinitialize parameters for each noise level

        # Add Gaussian noise directly to input data
        X_noisy = X + np.random.normal(0, noise_level, X.shape)
        
        # Train the network with noisy input data
        loss_history, prediction, Y = nn.train(X_noisy, Y, epochs, learning_rate)
        results[noise_level] = loss_history  # Store the loss history for analysis and plotting

        # Plot the loss curve for the current noise level
        plt.plot(loss_history, label=f'Noise Level {noise_level}')

    # Display all loss curves on the same plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Time for Different Noise Levels in Input Data')
    plt.legend()
    plt.show()

    # Print final predictions and target for analysis
    np.set_printoptions(precision=2, suppress=True)
    print("Final Predictions (for the last noise level tested):\n", prediction)
    print("Target:\n", Y)

    return results


if __name__ == "__main__":
    np.random.seed(42)
    nn = NeuralNetwork(8, 3, 8)
    nn.paramsInit()
    learningRate = 0.1
    X = np.eye(8)  # Identity matrix as input
    Y = X  # Target values

    epochs = 10000
    loss_history, prediction, y = nn.train(X, Y, epochs, learningRate)
    

    np.set_printoptions(precision=2, suppress=True)
    print("Final Predictions for default parameters:\n", prediction)
    print("Target:\n", Y)
    plt.plot(range(epochs), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over time (epochs)')
    plt.show()

    #Experiment 1: learning rate tweak
    #learning_rates = [0.001, 0.01, 0.1, 0.5, 0.8, 1]
    #learning_rate_experiment(nn, X, Y, epochs, learning_rates)

    # Experiment 2: weight initialization experiment
    #init_methods = ["uniform", "zero"]
    #results = weight_initialization_experiment(nn, X, Y, init_methods, epochs, learningRate)

    # Experiment 3: bias initialization experiment
    #bias_initializations = ["random", "zero", "high"]
    #results = bias_initialization_experiment(nn, X, Y, bias_initializations, epochs, learningRate)

    # Experiment 4: noisy input
    #noise_levels = [0.0, 0.1, 0.3, 0.5]
    #results = noise_experiment(nn, X, Y, noise_levels, epochs, learningRate)
