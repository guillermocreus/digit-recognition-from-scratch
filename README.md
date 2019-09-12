# Digit Recognition

This is an implementation of a neural network used for digit recognition, using the MNIST dataset. The implementation is done **from scratch**, in the sense that we are not using any existing machine learning libraries.
It was firstly developed as a two-layer neural network, but as of today it is fully customizable in the layer frame.

## First steps

The Git repository contains many files, ranging from binary classifiers to the neural network itself. In order to access the code containing the network, you should go to the folder *neural_network* and open the file *neural_net_class.py*.
The latest version just executes itself and starts the iterations. In order to finetune the model you should go to this piece of code:

```
def main():
    L = 4
    layer_dimensions = (L + 1)*[20]
    layer_dimensions[0] = 28 * 28 + 1
    layer_dimensions[-1] = 10

    factores = L * [1e-2]
    for i in range(L-1):
        factores[L-2-i] = factores[L-2-i+1] / 2

    activation_functions = L*[relu_v]
    activation_functions[-1] = softmax

    derivation_functions = L*[relu_d_v]
    derivation_functions[-1] = softmax

    NeuralNetwork = NeuralNet(L, layer_dimensions, factores,
                              activation_functions, derivation_functions)
    NeuralNetwork.gradient_descent(0.2, 4000, data_train, label_train,
                                   data_test, label_test)

```

## Structure and contents
An entire explanation and derivation of the project is kept entirely in the following [file](https://gitlab.com/guillermocreus98/digit-recognition/blob/master/NeuralNetwork_theory__1_.pdf).
This project was written entirely in [Python](https://www.python.org/) version 3.7. Several libraries from the standard library were used, and the latest version of the numpy library at current time, see [NumPy](https://numpy.org/).

## Authors

* **Guillermo Creus** - *Main contributor* - [guillermocreus98](https://github.com/PurpleBooth)
* **Victor Picón** - *Main contributor* - [vpicon](https://gitlab.com/vpicon)

## Final Note
Kwargs.
