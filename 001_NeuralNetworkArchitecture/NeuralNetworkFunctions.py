# Dependencies
import numpy as np



# Activation Function and Gradient Calculations

def sigmoid(x):
    """ Calculate the Sigmoid Activation Function over the output of a Neural Network"""
    return [1./(1.+np.exp(-x)) for xs in x]


def sigmoid_derivative(x):
    """ 
    Computes the Gradient of the sigmoid function with respect to its input x.
    """
    s = sigmoid(x)
    ds = [s*(1-s) for s in s]

    return ds

def softmax(x):
    """
    Calculates the Softmax Activation Function to scale the layer outputs to a range of 0-1
    with a sum of 1. This represents propability distributions over the length of x.
    """

    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


# Loss Functions

def L1(yhat, y):
    """
    yhat -- vector of size m (predicted_labels)
    y -- vector of size m (true labels)
    """

    return np.sum(abs(yhat-y))


def L2(yhat, y):
    """
    yhat -- vector of size m (predicted_labels)
    y -- vector of size m (true labels)
    """

    return np.sum(np.square(y-yhat))

# Array Actions
def image2vector(image):
    """
    Reshapes a given image into a colum vector representation
    """

    return image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))


def normalizeRows(x):
    """
    Implements each row of the given array to have unit length
    """

    # Compute x_norm
    x_norm = np.linalg.norm(x, 2, 1, True)

    # Divide x by its norm to rescale to unit length
    return x/x_norm


