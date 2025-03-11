# Multilayer Perceptron #

This repository contains code for the implementation of a multilayer perceptron
from scratch in Python. This is done using numpy and matplotlib, and involves
the implementation of neurons (in the form of weight matrices and bias 
vectors), layers, activation functions, and forward and backward propagation.

Currently, there is a bug with the softmax portion of this project, however
regression is relatively accurate when run against the provided miles per
gallon dataset.

### Usage ###

This project can be used by 
* Cloning its repository
* Installing pyenv for python version management, and setting 3.10.15 as the
  local python version
* Entering a virtual environment in the working directory
* Running `pip -r requirements.txt` in the working directory of the repository
* running `python3.10 ./train_mpg.py` in the working directory of the repository

This will show predictions for input values of the test set in the console,
as well as graphing validation and training loss over time in a window created
by Matplotlib.