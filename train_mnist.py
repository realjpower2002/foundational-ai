#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    
#
# Verify Reading Dataset via MnistDataloader class
#
import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = './mnist'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

val_size = int(len(x_train) * 0.2)

x_val, y_val = x_train[:val_size], y_train[:val_size]
x_train, y_train = x_train[val_size:], y_train[val_size:]

# Now you can use x_train, y_train, x_test, and y_test
print(f"x_train shape: {len(x_train)} samples")
print(f"y_train shape: {len(y_train)} labels")
print(f"x_val shape: {len(x_val)} samples")
print(f"y_val shape: {len(y_val)} labels")
print(f"x_test shape: {len(x_test)} samples")
print(f"y_test shape: {len(y_test)} labels")

# Function to flatten all images in a list
def flatten_images(image_list):
    flattened_images = []
    for image in image_list:
        flattened = []
        for line in image:
            flattened.extend(line)
        
        flattened_images.append(np.array(flattened))
    
    return np.array(flattened_images)



x_train = flatten_images(x_train)

x_val = flatten_images(x_val)

x_test = flatten_images(x_test)


print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

y_train = np.array(y_train)[...,np.newaxis]
y_val = np.array(y_val)[...,np.newaxis]
y_test = np.array(y_test)[...,np.newaxis]

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

import mlp

activations = mlp.Sigmoid
softmax = mlp.Softmax

multilayerperceptron = mlp.MultilayerPerceptron(
    (mlp.Layer(784, 1500, activations),
     mlp.Layer(1500,1500, activations),
     mlp.Layer(1500,1500, activations),
     mlp.Layer(1500,10, softmax))
)

ce = mlp.CrossEntropy


training_loss, validation_loss = multilayerperceptron.train(x_train, y_train, x_val, y_val, loss_func=ce, epochs = 10, learning_rate=0.001, batch_size=128)

y_pred = multilayerperceptron.forward(x_test)


y_pred_classes = y_pred
y_test_classes = y_test

print("y pred shape : ",y_pred_classes.shape)
print("y test shape : ",y_pred_classes.shape)

y_pred_classes = np.argmax(y_pred, axis=0)
y_test_classes = np.argmax(y_test, axis=0)

print(y_pred_classes[...,np.newaxis])
print(y_test_classes[...,np.newaxis])

accuracy = np.count_nonzero(y_pred_classes == y_test_classes)/y_test_classes.shape[0] * 100

print(f"Accuracy: {accuracy:.2f}%")