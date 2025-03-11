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
# print(f"Rows removed: {rows_removed}")

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

print(f"Samples in Training:   {len(X_train)}")
print(f"Samples in Validation: {len(X_val)}")
print(f"Samples in Testing:    {len(X_test)}")

import numpy as np 

x_train = X_train.to_numpy()
y_train = y_train.to_numpy()[...,np.newaxis] # y_train is 1D and needs to be 2D

x_val = X_val.to_numpy()
y_val = y_val.to_numpy()[...,np.newaxis]

import mlp

activation = mlp.Sigmoid
linear = mlp.Linear

multilayerperceptron = mlp.MultilayerPerceptron(
    (mlp.Layer(7, 30, activation),
     mlp.Layer(30, 30, activation),
     mlp.Layer(30, 30, activation),
     mlp.Layer(30,1, linear))
)

se = mlp.SquaredError



mlp.batch_generator(x_train, y_train, 4)



# print("Training Inputs : \n", x_train)
# print("Training Labels : \n", y_train.T)


training_loss, validation_loss = multilayerperceptron.train(x_train, y_train, x_val, y_val, loss_func=se, epochs = 100, learning_rate=0.001, batch_size=4)


x_test = X_test.to_numpy().transpose()
x_test_mean = np.mean(x_test, axis = 1, keepdims=True)
x_test_std = np.std(x_test, axis = 1, keepdims=True)
x_test = (x_test - x_test_mean) / x_test_std


y_test = y_test.to_numpy()[...,np.newaxis].transpose()


print(x_test.shape)
print(y_test.shape)


y_pred = multilayerperceptron.forward(x_test)

print("y_pred : \n",y_pred)
print("y_test : \n",y_test)
    
# This is essentially the average percent correct 
def mre(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)), np.mean(np.abs(y_true)), np.mean(np.abs(y_pred))

avg_mpg_off, avg_mpg_true, avg_mpg_pred = mre(y_test, y_pred)


print("Accuracy : ",avg_mpg_off, "mpg off on average.",avg_mpg_true,"avg mpg and",avg_mpg_pred,"avg predicted mpg.")