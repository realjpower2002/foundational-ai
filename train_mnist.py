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

# Function to flatten all images in a list
def flatten_images(image_list):
    flattened_images = []
    for image in image_list:
        flattened = []
        for line in image:
            flattened.extend(line)
        
        flattened_images.append(np.array(flattened))
    
    return np.array(flattened_images)



# Training and validation inputs are first flattened 
x_train = flatten_images(x_train)

x_val = flatten_images(x_val)

# Code for normalizing test images is borrowed from mpg
x_test = flatten_images(x_test).transpose()

# We obtain the mean and std dev across all samples for all features
x_test_mean = np.mean(x_test, axis = 1, keepdims=True)
x_test_std = np.std(x_test, axis = 1, keepdims=True)

# Prevents nan for 0 std dev
with np.errstate(divide='ignore', invalid='ignore'):
    x_test = (x_test - x_test_mean) / x_test_std
    x_test = np.nan_to_num(x_test, nan=0.0)


# Produces y array from vector
y_train = np.array(y_train)[...,np.newaxis]
y_val = np.array(y_val)[...,np.newaxis]

# Transpose this as well just to make sure dimensions
# work (forward requires a column vector, whereas
# train requires each new sample is on its own Row.)
y_test = np.array(y_test)[...,np.newaxis].transpose()



import mlp

# Define activations and loss function

activations = mlp.Sigmoid
softmax = mlp.Softmax

multilayerperceptron = mlp.MultilayerPerceptron(
    (mlp.Layer(784, 1500, activations),
     mlp.Layer(1500,1500, activations),
     mlp.Layer(1500,1500, activations),
     mlp.Layer(1500,10, softmax))
)

ce = mlp.CrossEntropy

# Train model

training_loss, validation_loss = multilayerperceptron.train(x_train, y_train, x_val, y_val, loss_func=ce, epochs = 16, learning_rate=0.001, batch_size=512)

# Predict classes for test input x_test

y_pred = multilayerperceptron.forward(x_test)

# Classes are then determined by obtaining the index in the output with the
# greatest value

y_pred_classes = y_pred
y_test_classes = y_test

y_pred_classes = np.argmax(y_pred, axis=0)
y_test_classes = np.argmax(y_test, axis=0)

# print(" PRED ")
# for i, y_class in enumerate(y_pred_classes):
#     print(y_class, sep = ' ')
#     if i > 15:
#         break

# print(" TEST ")

# for i, y_class in enumerate(y_test_classes):
#     print(y_class, sep = ' ')
#     if i > 15:
#         break

# Accuracy is determined empirically and reported

accuracy = np.count_nonzero(y_pred_classes == y_test_classes)/y_test_classes.shape[0] * 100

print(f"Accuracy: {accuracy:.2f}%")