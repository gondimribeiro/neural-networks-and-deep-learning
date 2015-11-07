import json
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
import theano
import theano.tensor as T
import random
import numpy as np

def ReLU(z): return T.maximum(0.0, z)

training_data, validation_data, test_data = network3.load_data_shared()
#training_data, validation_data, test_data = network3.load_data_shared('../data/mnist_expanded.pkl.gz')

# Hyperparamters
mini_batch_size = 10
num_epochs = 30
eta = 0.03
filename = "data"
lmbda = 0
p_dropout = 0.16666666666667

# Set seed to facilitate reproducibility
random.seed(12345678)
np.random.seed(12345678)

# Build network
net = Network(
    [
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(20, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=20*4*4, n_out=30, activation_fn=ReLU, p_dropout=p_dropout),
        FullyConnectedLayer(n_in=30, n_out=30, activation_fn=ReLU, p_dropout=p_dropout), 
        SoftmaxLayer(n_in=30, n_out=10, p_dropout=p_dropout)
    ],
    mini_batch_size)

# Call SGD
training_accuracy, validation_accuracy = net.SGD(
    training_data,
    num_epochs,
    mini_batch_size,
    eta,
    validation_data,
    test_data,
    lmbda=lmbda,
    monitor_data=True)

# Save data on file
f = open(filename + ".txt", "w")
json.dump([training_accuracy, validation_accuracy], f)
f.close();
