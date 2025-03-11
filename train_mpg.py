import pandas as pd
import os
import pickle

# (I was getting rate limited during testing)
file_name = "mpg.pickle"
file_path = os.path.join(os.getcwd(), file_name)

with open(file_path, "rb") as file:
    data = pickle.load(file)

# Drop rows where the target variable is NaN
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

# Display the number of rows removed
rows_removed = len(data) - len(cleaned_data)

from sklearn.model_selection import train_test_split

# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,    # for reproducibility
    shuffle=True,       # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

# Convert training and validation sets to numpy arrays with 
# each sample on its own row

import numpy as np 

x_train = X_train.to_numpy()
y_train = y_train.to_numpy()[...,np.newaxis] # y_train is 1D and needs to be 2D

x_val = X_val.to_numpy()
y_val = y_val.to_numpy()[...,np.newaxis]



import mlp

# Activations and loss functions are defined

activation = mlp.Tanh
linear = mlp.Linear

multilayerperceptron = mlp.MultilayerPerceptron(
    (mlp.Layer(7, 30, activation),
     mlp.Layer(30, 30, activation),
     mlp.Layer(30, 30, activation),
     mlp.Layer(30,1, linear))
)

se = mlp.SquaredError

# Define number of epochs
num_epochs = 200

# Train model

training_loss, validation_loss = multilayerperceptron.train(x_train, y_train, x_val, y_val, loss_func=se, epochs = num_epochs, learning_rate=0.001, batch_size=4)


# Transpose test set input and output for forwarding

x_test = X_test.to_numpy().transpose()
x_test_mean = np.mean(x_test, axis = 1, keepdims=True)
x_test_std = np.std(x_test, axis = 1, keepdims=True)
x_test = (x_test - x_test_mean) / x_test_std

y_test = y_test.to_numpy()[...,np.newaxis].transpose()


# Produce predictions 

y_pred = multilayerperceptron.forward(x_test)
    


# This is essentially the average percent correct 
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true) * 100, np.mean(np.abs(y_true - y_pred)), np.mean(np.abs(y_true)), np.mean(np.abs(y_pred))

mae_score, avg_mpg_off, avg_mpg_true, avg_mpg_pred = mae(y_test, y_pred)

print("Predicted Y:",y_pred)
print("True Y:",y_test)

print("Accuracy:",mae_score, "(MAE Score).", avg_mpg_off, "mpg off on average.",avg_mpg_true,"avg mpg and",avg_mpg_pred,"avg predicted mpg.")

loss = se.loss(se, y_true = y_test, y_pred = y_pred)
print("Total Test Loss:",loss)

# Generate matplotlib plots
import matplotlib.pyplot as plt

num_epochs = range(1,num_epochs+1)

# Create the plot
plt.plot(num_epochs, training_loss, label='Training Loss', color='blue', linestyle='-', marker='o')
plt.plot(num_epochs, validation_loss, label='Validation Loss', color='red', linestyle='-', marker='x')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')

# Add a legend
plt.legend()

# Display the grid
plt.grid(True)

# Display the plot window
plt.show()  # This is the command that opens the plot window