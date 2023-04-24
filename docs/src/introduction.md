# Introduction

This code is an implementation of a neural network for handwritten digit recognition using
the MNIST database. It is written in the Julia programming language.

## Overview

The code defines a simple neural network with a customizable number of layers and neurons
per layer, activation function, and its derivative. The network is trained using stochastic
gradient descent with mini-batches.

### Data Structures

- `Example`: A structure containing input and output data
- `Network`: A structure representing the neural network
- `Estimator`: A structure to evaluate the network's performance on given data
- `EachLayer`: An iterator to traverse layers of the network
- `Backpropagator`: A structure to compute gradients for weight and bias updates
- `sigmoid`, `sigmoid'`: Activation function and its derivative

### Training Functions

- `train!`: Train the neural network using stochastic gradient descent with mini-batches
- `train`: Update weights and biases of the network for a single example
- `feedforward`: Perform a forward pass through the network

### Data Loading

- `loaddata`: Load training or testing data from the MNIST dataset

## Usage

1. Create a neural network with the desired number of layers and neurons per layer.

   ```julia
   network = Network(784, 30, 10)
   ```

2. Load the training data.

   ```julia
   train_data = loaddata(:train)
   ```

3. Train the neural network using the training data.

   ```julia
   train!(network, train_data, batchsize=10, η=3, nepochs=30)
   ```

4. Load the test data.

   ```julia
   test_data = loaddata(:test)
   ```

5. Evaluate the network's performance on the test data.

   ```julia
   estimator = Estimator(network, sigmoid)
   result = estimator(test_data)
   println("Accuracy: ", result.hits)
   println("Loss: ", result.loss)
   ```

## Code Explanation

- The code starts by importing necessary packages and defining the main data structures.
- The `Network` structure is defined using a computed field type to store the layer sizes,
  weights, and biases.
- The `feedforward` function takes an activation function, the network's weights and biases,
  and an input vector. It computes the output of the network by applying the activation
  function on each layer.
- The `Estimator` structure is used to evaluate the performance of the network on a given
  dataset.
- The `EachLayer` iterator is used to traverse the layers of the network.
- The `Backpropagator` structure computes the gradients for weight and bias updates using
  the backpropagation algorithm.
- The `train!` function is the main training function that performs stochastic gradient
  descent with mini-batches. It updates the weights and biases of the network.
- The `loaddata` function loads the MNIST dataset and preprocesses it for training.

Please refer to the comments in the code for a more detailed explanation of each function
and data structure.
