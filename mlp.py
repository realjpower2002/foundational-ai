import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple
import math
import random

def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) 
    and batch_y has shape (B, q). The last batch may be smaller.
    """

    # print("Train x (batch) : ",np.shape(train_x))
    # print("Train y (batch) : ",np.shape(train_y))

    # print("Train y batch : ",train_y)

    # Normalizing values : 
    print(np.mean(train_x, axis = 0))
    print(np.std(train_x, axis = 0))

    # Beginning, normalization step
    #
    # Start as one sample per row, so we are averaging along rows
    with np.errstate(divide='ignore', invalid='ignore'):
        train_x = (train_x - np.mean(train_x, axis = 0)) / (np.std(train_x, axis = 0))

        # print("Mean across samples : ",np.shape(np.mean(train_x, axis = 0)))
        # print("std dev across samples : ",np.shape(np.std(train_x, axis = 0)))

        train_x = np.nan_to_num(train_x, nan=0.0)


    # Train x : 
    # print("NEW TRAIN X !!!!! : ",train_x[0])

    # New arrays that will be created
    train_x_batches = []
    train_y_batches = []

    # Number of batches we will use for training. This is used
    # to get the index for producing the training batches.
    num_train_batches = math.ceil(train_x.shape[0] / batch_size)

    i = 0

    for batch_index in range(num_train_batches):

        train_x_batch = []
        train_y_batch = []

        # Add each individual sample to the n x f or n x q sample
        for sample_number in range(batch_size):
            train_index = batch_index * batch_size + sample_number

            # Short circuit to prevent out of bounds indexing
            if train_index >= len(train_x):
                break
        
            # Append a length-784 row to the batch
            train_x_batch.append(train_x[train_index])

            train_y_batch.append(train_y[train_index])

        if i == 0:
            # print("Train x batch 1 : ",np.shape(train_x_batch))
            # print("Train y batch 1 : ",np.shape(train_y_batch))
            i+=1

        train_x_batches.append(np.array(train_x_batch))
        train_y_batches.append(np.array(train_y_batch))

    # print("Train x batches : ",np.shape(train_x_batches))
    # print("Train y batches : ",np.shape(train_y_batches))

    # Return np array of numpy arrays
    return train_x_batches, train_y_batches
        


    


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:


        # print(type(x))
        # print(x.dtype)
        # print("Sigmoid Input : ",x)
        # # Define sigmoid using numpy functions
        # print("Sigmoid Activation : ",(1 / (1 + np.exp(-x))))
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:

        # Define sigmoid
        sigmoid = 1 /(1 + np.exp(-x))

        # Define derivative of sigmoid using the sigmoid definition
        return (sigmoid * (1 - sigmoid))



class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:

        # Define sinh and cosh
        sinh = (np.exp(x)-np.exp(-x)) / 2
        cosh = (np.exp(x)+np.exp(-x)) / 2

        # Define tanh by dividing the two
        return sinh/cosh

    def derivative(self, x: np.ndarray) -> np.ndarray:

        # Define sinh and cosh
        sinh = (np.exp(x)-np.exp(-x)) / 2
        cosh = (np.exp(x)+np.exp(-x)) / 2

        # Define tanh by dividing the two
        tanh = sinh/cosh

        # Define derivative of tanh as 1 - tanh^2
        return 1 - np.pow(tanh,2)



class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:

        # Implement ReLU in this stupid simple way
        return np.maximum(x,0)

    def derivative(self, x: np.ndarray) -> np.ndarray:

        # ReLU derivative is also very simple.
        return np.where(x<0, 0, 1)


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:

        # get exponents of everything
        exps = np.exp(x)

        # compute column sums
        summed_exps = np.sum(exps,axis=0)

        # divide columsn by column sums
        return exps / summed_exps
    
    def derivative(self, x: np.ndarray) -> np.ndarray:

        # Get softmax activations for x (linear output from 
        # weights and biases of a given layer)
        s = self.forward(self, x)

        # Turn s into a 3D array, with each column being a vector
        s = s[...,np.newaxis]

        jacobians = []

        # Iterate over columns, to get the full Jacobian for each
        # of the samples. This is not good practice (and isn't
        # vectorized), but this isn't used in the main training
        # loop anyway.
        for sample in np.transpose(s, (1,0,2)):

            # Make identity matrix (mask for derivative) with width and height n
            main_diag = np.eye(sample.shape[0])

            # I want to figure out how Dr. Ghawaly did this with einsum ...
            #
            # bij, bj -> bi

            # Define off-diagonal matrix
            off_diagonal = np.dot(sample, -sample.T)
            
            # Define on-diagonal array by doing arithmetic and flattening
            # ndarray 'x' before converting this 1D array into a diagonal
            # matrix.
            on_diagonal = np.diag(sample * (1 - sample).flatten())

            # Choose (-si * sj) where i is the row and j is the column,
            # on off-diagonal
            #
            # Choose (si * (1 - si)) where i is the row on on-diagonal
            #
            jacobian = np.where(main_diag == 1, on_diagonal, off_diagonal)

            # Append this to the list of jacobians
            jacobians.append(jacobian)

        jacobians = np.stack(jacobians, axis = 0)

        return jacobians


class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:

        # Simply return x
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:

        # Literally just return all 1s, as a diagonal matrix
        return np.ones_like(x)


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

        return np.sum(np.pow(y_true - y_pred,2)) / (2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

        return (y_pred-y_true)



class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

        return -np.sum(y_true * np.log(y_pred))

    # Suspiciously simple ...
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

        return y_pred - y_true


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        # Initialize weights and biases vectors
        glorot_unit = math.sqrt(6/(fan_in+fan_out))

        # Transpose to make rows go down and columns go out
        self.W = np.array([[(random.random() * glorot_unit * 1 - glorot_unit * 0.5) 
                                          for w_cols in range(fan_in)] 
                                          for w_rows in range(fan_out)])
        
        # print("Initial Weights\n",self.W)
                    
        # Transpose to make array vertical
        self.b = np.ones((fan_out,1)) * 0.1

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """

        # Get dot products of current weights (fan_out x fan_in) 
        # and input (fan_in x 1) to get new pre activation vector 
        # (fan_out x 1)

        # print("Weight dimensions : ",np.shape(self.W))
        # print("Input dimensions : ",np.shape(h))

        z = np.dot(self.W, h) + self.b

        # print("Pre activation dimensions : ",np.shape(z))
        # print("Pre-Activation : ",z,"\n    (Shape : ",z.shape,")")

        # print("Pre activation : ")
        # print(z)

        # print("Z after combination : ",z)

        # Get output from activation function
        phi = self.activation_function.forward(self.activation_function, z)

        # print("Activation : ",phi,"\n    (Shape : ",phi.shape,")")

        self.activations = phi

        # print("Activation : ")

        # Easy money
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """

        # Just keeping things straight in my head ...
        #
        # The Loss - This derivative is always a matrix where each column is a sample and each row is the
        # loss produced by one of the output neurons.
        dL_dA = delta 

        # Compute pre-activation
        z = np.dot(self.W, h) + self.b

        # print("W shape : ",self.W.shape)
        # print("h shape : ",h.shape)
        # print("Pre-activations : ",z)

        #print("Backpropagating")
        
        # region GET dL/dZ

        # For the output layer of a multiclass classifier specifically, 
        # dL/dZ is the same as y_true-y_pred. This skips the extremely 
        # complicated and somewhat compute intensive step of calculating 
        # dozens of full jacobians per batch.
        if(self.activation_function == Softmax):
            y_pred = self.activation_function.forward(self.activation_function, z)

            # print("Predictions reduced from the pre-activations x (z) : ",y_pred)

            # Get y_true back by doing the reverse of the activation function
            # y_true = -dL_dA + y_pred

            # print("Real y values : ",y_true)

            dL_dZ = dL_dA

            # print("dL_dZ (softmax) : ",dL_dZ)

        else:
            # Get derivative of activation function for this pre-activation
            dA_dZ = self.activation_function.derivative(self.activation_function, z)

            # print("dL_dA : ", dL_dA)
            # print("dA_dZ : ",dA_dZ)

            dL_dZ = dL_dA * dA_dZ
            # print("dL_dZ (NOT softmax) : ",dL_dZ)

        # endregion

        # Simple explanation of dZ_dW and dZ_db
        dZ_dW = np.transpose(h)
        dZ_db = 1

        # Calculate dL_dW and dL_db

        # print("Hadamarding dL_dA and dA_dZ : ")
        # Have to also divide by batch size to prevent gradient from exploding immediately 

        dL_dW = np.dot(dL_dZ, dZ_dW) / dL_dZ.shape[1]
        dL_db = np.sum(dL_dZ * dZ_db, axis=1, keepdims = True) / dL_dZ.shape[1]

        # print("Change L wrt W")
        # print(dL_dW)
        # print("Change L wrt b")
        # print(dL_db)

        # Calculate new delta
        self.delta = np.dot(np.transpose(self.W) , (dL_dZ))

        # print("Got dL_dW : ",dL_dW.shape)
        # print("Got dL_db : ",dL_db.shape)

        # print("dL/dW : ",dL_dW,"\n    (Shape : ",dL_dW.shape,")")
        # print("dL/db : ",dL_db,"\n    (Shape : ",dL_db.shape,")")

        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """

        # Not really much to it...

        for layer in self.layers:

            x = layer.forward(x)

        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """

        # Initialize empty dl_dw and dl_db lists
        dl_dw_all = []
        dl_db_all = []

        delta = loss_grad

        # Go through layers in reverse calculating the new delta as we go.
        for layer_index in range(len(self.layers)-1,-1,-1):

            # print("backpropping layer",layer_index)

            # Get the current layer
            layer = self.layers[layer_index]

            # Get the layer input
            if(layer_index == 0):
                layer_input = input_data
            else:
                layer_input = self.layers[layer_index-1].activations

            # Get this layer's dL/dW and dL/db
            dL_dW, dL_db = layer.backward(layer_input, delta)

            dl_dw_all.insert(0,dL_dW)
            dl_db_all.insert(0,dL_db)

            # Next delta passed will be the one produced by this layer
            delta = self.layers[layer_index].delta


        # print("Got dl_dw_all : ",dl_dw_all)
        # for dl_dw in dl_dw_all:
        #     print(dl_dw.shape)

        return dl_dw_all, dl_db_all
    

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (d x n) n = number of samples, d = number of features
        :param train_y: full training set output of shape (q x n) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """

        # Generate all batches for training set
        train_x_batches, train_y_batches = batch_generator(train_x, train_y, batch_size)

        # Generate single large "batch" for validation set. This is done
        # to use the same normalization on validation as on train.
        val_x, val_y = batch_generator(val_x, val_y, np.shape(val_x)[0])

        # Convert validation x and y to proper dimensions (rows = features, 
        # columns = samples)
        val_x = val_x[0].T
        val_y = val_y[0].T

        # print("Train x : ",np.shape(train_x_batches))
        # print("Train y : ",np.shape(train_y_batches))

        # print("Validation X Shape : ",val_x.shape)

        # Keep training and validation losses to return later.
        training_losses = []
        validation_losses = []

        # print("Length of batches list : ",len(train_x_batches))

        for epoch in range(epochs):

            # new_learning_rate = 0.0001 + (0.03 - 0.0001) * np.exp(-0.005 * epoch)

            # Keep track of total loss values for all batches together
            training_loss = 0 

            for batch_num in range(len(train_x_batches)):

                # I do this because my model uses rows as features and columns as samples,
                # instead of vice versa for the data input into the model
                batch = train_x_batches[batch_num].T

                # print(np.shape(batch))

                truth = train_y_batches[batch_num].T

                


                # Refresh deltas and activations through forward propagation first
                predictions = self.forward(batch)

                # print("Predictions : ",predictions)
                # print("Truth : ",truth)
                # print("Mean prediction : ",np.mean(predictions))
                # print("Activation of layer 2 : \n",self.layers[1].activations)

                # print("Standard Deviation Predictions : ",np.std(predictions))

                # Get loss gradient from truths and predictions
                loss_gradient = loss_func.derivative(loss_func, truth, predictions)

                # print("Loss Gradient : ",loss_gradient)

                # Get weights and biases gradients for each layer during backpropagation
                dl_dw_all, dl_db_all = self.backward(loss_gradient, batch)

                # print("dL/dW : \n",dl_dw_all[1])

                # print("dL/dW for all layers : ",dl_dw_all)

                # Update weights and biases
                for layer_index in range(len(self.layers)):

                    layer = self.layers[layer_index]

                    # print(dl_dw_all[layer_index].shape)
                    # print(dl_db_all[layer_index].shape)

                    # Update weights and biases using weights and biases gradients
                    # print("DL/dW Shape : ",dl_dw_all[layer_index])
                    # print(dl_db_all[layer_index])

                    # print("dl_dw : ", dl_dw_all[layer_index])
                    # print("dl_db : ", dl_db_all[layer_index])

                    layer.W = layer.W - learning_rate * dl_dw_all[layer_index]
                    layer.b = layer.b - learning_rate * dl_db_all[layer_index]

                # Get new predictions (after updating weights and biases)
                predictions = self.forward(batch)
                training_loss += loss_func.loss(loss_func, truth, predictions)

            training_loss /= len(train_x_batches)

            validation_predictions = self.forward(val_x)
            # print("Validation Predictions : ",validation_predictions)

            validation_loss = np.sum(loss_func.loss(loss_func, val_y, validation_predictions))

            print("Epoch",epoch,": Training loss is",training_loss,", Validation loss is",validation_loss,". Learning Rate is",learning_rate)

            training_losses.append(training_loss)
            validation_losses.append(validation_loss)

        return training_losses, validation_losses