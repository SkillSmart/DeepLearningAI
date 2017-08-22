# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


# Define Helper Functions
def train_test_split(dataset, testsize=0.9):
    """
    Generates a split in train and test split for both targets and labels
    given a dataset
    """
    X_train, X_test = dataset[0][ :int(len(dataset[0])*testsize)], dataset[0][int(len(dataset[0])*testsize): ]
    y_train, y_test = dataset[1][ :int(len(dataset[1])*testsize)], dataset[1][int(len(dataset[1])*testsize): ] 

    return X_train, X_test, y_train, y_test

# Load the Dataset and Create a Train Test split
dataset = np.load('XXXXX.pkl')
X_train, X_test, y_train, y_test = train_test_split(dataset)


# Flatten the Train and testset
""" Uses the Transpose on the reshape """
X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T


# Create the Model

# Initialize the parameters



# Forward & Backward Propagation
def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)                                     
    cost = -1/m * np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost