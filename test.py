import mlp
import numpy as np

x = np.array([0.1, 0.5, -0.7, 1.0, -0.3])
x = x[...,np.newaxis]

print("Input : ",x)

# print("SIGMOID : ")

sigmoid = mlp.Sigmoid

# print(sigmoid.forward(sigmoid, x=x))
# print(sigmoid.derivative(sigmoid, x=x))

# print("TANH : ")

# tanh = mlp.Tanh

# print(tanh.forward(tanh, x=x))
# print(tanh.derivative(tanh, x=x))

# print("RELU : ")

# relu = mlp.Relu

# print(relu.forward(relu, x=x))
# print(relu.derivative(relu, x=x))

# print("SOFTMAX :")

softmax = mlp.Softmax

# print(softmax.forward(softmax, x=x))
# print(softmax.derivative(softmax, x=x))

# print("LINEAR :")

# linear = mlp.Linear

# print(linear.forward(linear, x=x))
# print(linear.derivative(linear, x=x))

print("Making a Layer (5 neurons -> 7 neurons):")

layer = mlp.Layer(5, 7, sigmoid)

print("Weights : ")
print(layer.W)

print("Running forward prop : ")

forwarded = layer.forward(x)

print(forwarded)

print("Running backprop : ")

print(layer.backward(x, forwarded))

cross_entropy = mlp.CrossEntropy

data = np.array([[-0.2, 1.5, -0.1, 2.2, 0.8],[0.1,-0.9,2.1,-0.3,-0.9]])

data = data.transpose()


print("\n\n\nGetting derivative of softmax")

print(data)
print(data.shape)

print(softmax.derivative(softmax, data))


print("Testing backprop of softmax : ")
layer = mlp.Layer(5,7,softmax)

# x = np.array([0.1, 0.5, -0.7, 1.0, -0.3]).reshape(-1,1)

forwarded = layer.forward(data)

print("Forward prop : \n",forwarded)

y_pred = forwarded
y_true = np.array([[0, 0, 0, 0, 0, 1, 0],[0,1,0,0,0,0,0]]).T

print("Predictions shape : ",y_pred.shape)
print("Truths shape : ",y_true.shape)

delta = cross_entropy.derivative(cross_entropy, y_true, y_pred)

print("Original y true\n",delta * -y_pred)

print("delta : ",delta)

print("Backprop : dL/dW and dL/db : \n", layer.backward(data, delta))






mse = mlp.SquaredError
layer = mlp.Layer(5,7,sigmoid)

print("\n\n\n\n\n\n\n\nTesting backprop of sigmoid : ")

print(data)

print("Derivative")
print(sigmoid.derivative(sigmoid, data))

print("Testing backprop ...")

forwarded = layer.forward(data)

print("Forward prop : \n",forwarded)

y_pred = forwarded
y_true = np.array([[0.8, -0.7, 0.1, 0.1, 0.5, 0.6, -0.1],[0.3,0.5,0.6,0.3,0-0.8,-0.8,0.2]]).T

print("Predictions shape : ",y_pred.shape)
print("Truths : ",y_true.shape)

delta = mse.derivative(mse, y_true, y_pred)

# print("Original y true\n",delta * -y_pred)

print("delta : ",delta)

print("Backprop : dL/dW and dL/db : \n", layer.backward(data, delta))