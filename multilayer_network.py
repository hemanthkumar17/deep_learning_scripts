
# partial credit to tan_nguyen for the 3-layer network base code building this script
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.load_iris(return_X_y=True)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''
        y = None

        # YOU IMPLMENT YOUR actFun HERE
        if type == "Tanh":
            y = np.tanh(z)
        elif type == "Sigmoid":
            y = 1./(1. + np.exp(-z))
        elif type == "ReLU":
            y = z * (z > 0)
        return y

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''
        # YOU IMPLEMENT YOUR diff_actFun HERE
        dy = None
        if type == "Tanh":
            dy = 1 - self.actFun(z, type="Tanh")**2
        elif type == "Sigmoid":
            dy = self.actFun(z, type="Sigmoid") *(1 - self.actFun(z, type="Sigmoid"))
        elif type == "ReLU":
            dy = 1 * (z > 0)
        return dy

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = actFun(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.probs = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        data_loss = -1/self.nn_input_dim * np.sum(np.sum(np.eye(self.nn_output_dim)[y] * np.log(self.probs)))

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        del2 = self.probs
        del2[range(len(X)), y] -= 1

        # del2 = (y * (1/self.probs)) * (self.diff_actFun(self.z2, type="Sigmoid"))
        dW2 = self.a1.T.dot(del2)
        db2 = np.sum(del2)

        del1 = del2.dot(self.W2.T) * self.diff_actFun(self.z1, type=self.actFun_type)
        dW1 = X.T.dot(del1)
        db1 = np.sum(del1)

        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.001, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)
    
    
class DeepNeuralNetwork(NeuralNetwork) :
    """
    This class builds and trains a deep neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_layers, nn_hidden_layer_size, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_layers: number of hidden layers
        :param nn_hidden_layer_size: the number of hidden units in each layer
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim

        self.nn_hidden_layers = nn_hidden_layers
        self.nn_hidden_dim = nn_hidden_layer_size
        
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        # self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        # self.b1 = np.zeros((1, self.nn_hidden_dim))
        # self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        # self.b2 = np.zeros((1, self.nn_output_dim))
        self.params = {}
        self.params["W1"] = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.params["b1"] = np.zeros((1, self.nn_hidden_dim))
        
        for i in range(2,nn_hidden_layers+1):
            self.params[f"W{i}"] = np.random.randn(self.nn_hidden_dim, self.nn_hidden_dim) / np.sqrt(self.nn_hidden_dim)
            self.params[f"b{i}"] = np.zeros((1, self.nn_hidden_dim))
    
        self.params[f"W{nn_hidden_layers+1}"] = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.params[f"b{nn_hidden_layers+1}"] = np.zeros((1, self.nn_output_dim))
    
    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        # self.z1 = X.dot(self.W1) + self.b1
        # self.a1 = actFun(self.z1)
        self.params["z1"] = X.dot(self.params["W1"]) + self.params["b1"]
        self.params["a1"] = actFun(self.params["z1"])
        
        for i in range(2,self.nn_hidden_layers+1):
            self.params[f"z{i}"] = self.params[f"z{i-1}"].dot(self.params[f"W{i}"]) + self.params[f"b{i}"]
            self.params[f"a{i}"] = actFun(self.params[f"z{i}"])

        self.params[f"z{self.nn_hidden_layers+1}"] = self.params[f"a{self.nn_hidden_layers}"].dot(self.params[f"W{self.nn_hidden_layers+1}"]) + self.params[f"b{self.nn_hidden_layers+1}"]
        self.probs = np.exp(self.params[f"z{self.nn_hidden_layers+1}"]) / np.sum(np.exp(self.params[f"z{self.nn_hidden_layers+1}"]), axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        data_loss = -1/self.nn_input_dim * np.sum(np.sum(np.eye(self.nn_output_dim)[y] * np.log(self.probs)))

        # Add regulatization term to loss (optional)
        # data_loss += self.reg_lambda / 2 * (np.sum(np.sum([np.square(self.params[f"W{w}"]) for w in range(1, self.nn_hidden_layers+2)])))
        return (1. / num_examples) * data_loss

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: None
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        self.params[f"del{self.nn_hidden_layers+1}"] = self.probs
        self.params[f"del{self.nn_hidden_layers+1}"][range(len(X)), y] -= 1

        # del2 = (y * (1/self.probs)) * (self.diff_actFun(self.z2, type="Sigmoid"))
        # dW2 = self.a1.T.dot(del2)
        # db2 = np.sum(del2)

        self.params[f"dW{self.nn_hidden_layers+1}"] = self.params[f"a{self.nn_hidden_layers}"].T.dot(self.params[f"del{self.nn_hidden_layers+1}"])
        self.params[f"db{self.nn_hidden_layers+1}"] = np.sum(self.params[f"del{self.nn_hidden_layers+1}"])


        for i in range(self.nn_hidden_layers, 1, -1):
            self.params[f"del{i}"] = self.params[f"del{i+1}"].dot(self.params[f"W{i+1}"].T) * self.diff_actFun(self.params[f"z{i}"], type=self.actFun_type)
            self.params[f"dW{i}"] = self.params[f"a{i-1}"].T.dot(self.params[f"del{i}"])
            self.params[f"db{i}"] = np.sum(self.params[f"del{i}"])
        
        self.params[f"del1"] = self.params[f"del2"].dot(self.params[f"W2"].T) * self.diff_actFun(self.params[f"z1"], type=self.actFun_type)
        self.params[f"dW1"] = X.T.dot(self.params[f"del1"])
        self.params[f"db1"] = np.sum(self.params[f"del1"])
    
        # del1 = del2.dot(self.W2.T) * self.diff_actFun(self.z1, type=self.actFun_type)
        # dW1 = X.T.dot(del1)
        # db1 = np.sum(del1)

        return None

    def fit_model(self, X, y, epsilon=0.001, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            # dW1, dW2, db1, db2 = self.backprop(X, y)
            self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            # dW2 += self.reg_lambda * self.W2
            # dW1 += self.reg_lambda * self.W1
            for j in range(1, self.nn_hidden_layers+2):
                self.params[f"dW{j}"] += self.reg_lambda * self.params[f"W{j}"]
            # Gradient descent parameter update
            # self.W1 += -epsilon * dW1
            # self.b1 += -epsilon * db1
            # self.W2 += -epsilon * dW2
            # self.b2 += -epsilon * db2
            for j in range(1, self.nn_hidden_layers+2):
                self.params[f"W{j}"] += -epsilon * self.params[f"dW{j}"]
                self.params[f"b{j}"] += -epsilon * self.params[f"db{j}"]
                
            
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 100 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

def main():
    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    
    eps = 0.006
    nn_hidden_layer_size = 3
    nn_hidden_layers = 2
    print("^"*100)
    
    print("Number of hidden units: " + str(nn_hidden_layer_size))
    print("Number of hidden layers: " + str(nn_hidden_layers))
    print("Learning rate: "+ str(eps))
    print("^"*100)
    model = DeepNeuralNetwork(nn_input_dim=4, nn_hidden_layer_size=nn_hidden_layer_size, nn_hidden_layers=nn_hidden_layers , nn_output_dim=3, actFun_type="Sigmoid")
    model.fit_model(X,y, epsilon=eps)
    # model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()