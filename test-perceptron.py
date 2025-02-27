import mlp
import numpy as np


data = np.array([[-0.2, 1.5, -0.1, 2.2, 0.8],[0.1,-0.9,2.1,-0.3,-0.9]])

data = data.transpose()

relu = mlp.Relu
linear = mlp.Linear

layer1 = mlp.Layer(5,30, relu)
layer2 = mlp.Layer(30,30, relu)
layer3 = mlp.Layer(30,10, linear)

multilayer_perceptron = mlp.MultilayerPerceptron((layer1,layer2,layer3))

print(multilayer_perceptron.forward(data))

