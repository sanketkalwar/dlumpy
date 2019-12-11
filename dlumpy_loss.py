import numpy as np
import matplotlib.pyplot as plt

class Softmax:
    """[summary]
    x = np.random.randn(1,100)
    y = np.random.randn(1,100)
    softmax = Softmax(x)
    pred = softmax.forward()
    Returns:
        [type] -- [description]
    """
    def __init__(self,input_array=None):
        self.input_array = input_array
        self.output = None
        self.d_softmax = None

    def __del__(self):
        del self.input_array
        del self.output
        del self.d_softmax

    def forward(self):
        z = np.exp(self.input_array-np.max(self.input_array))
        self.output = z/np.sum(z)
        return self.output

class CrossEntropyLoss:
    """[summary]
    x = np.random.randn(1,100)
    y = np.random.randn(1,100)
    softmax = Softmax(x)
    pred = softmax.forward()
    cel = CrossEntropyLoss(predicted_output=pred.T,actual_output=x)
    out = cel.forward()
    print(out)

    Returns:
        [type] -- [description]
    """
    def __init__(self,predicted_output=None,actual_output=None):
        self.predicted_output = predicted_output
        self.actual_output = actual_output
        self.output = None
        self.d_ce   = None

    def __del__(self):
        del self.predicted_output
        del self.actual_output
        del self.output
        del self.d_ce

    def forward(self):
        log_prob = -1*np.log(self.predicted_output)
        self.output = np.dot(self.actual_output,log_prob)
        return self.output
