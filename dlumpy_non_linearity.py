import numpy as np

class Sigmoid:
    def __init__(self,input_array):
        self.input_array = input_array
        self.output = None
        self.d_sigmoid = None

    def __del__(self):
        del self.input_array
        del self.output
        del self.d_sigmoid

    def forward(self):
        self.output = 1./(1.+np.exp(self.input_array))
        return self.output

    def backward(self):
        self.d_sigmoid = self.input_array*(1-self.input_array)
        return self.d_sigmoid




def sigmoid(x):
    return 1./(1.+np.exp(x))

def tanh(x):
    return np.tanh(x)

def Relu(x):
    return np.maximum(x,1e-2)
