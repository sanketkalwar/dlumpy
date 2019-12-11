import cv2
import numpy as np
import matplotlib.pyplot as plt
from dlumpy_non_linearity import Relu,sigmoid,tanh
from dlumpy_loss import Softmax,CrossEntropyLoss

class Conv2D:
    """[Summary]
        size = 100
        temp  = np.random.rand(1,size,size,3)
        # image = cv2.imread('index.jpeg')
        # image = cv2.resize(image,(size,size))
        # temp[0,:,:,:] = image
        image = temp
        conv1  = Conv2D(input_array=image,kernel_size=3,out_channel=10,stride=1,pad_width=None)
        out1 = tanh(conv1.forward())
        conv2  = Conv2D(input_array=out1,kernel_size=3,out_channel=3,stride=1,pad_width=None)
        out2 = tanh(conv2.forward())
        print('input size',image.shape)
        print('output size',out2.shape)
        for i in range(out2.shape[0]):
            for j in range(out2.shape[3]):
                plt.imshow(out2[i,:,:,j])
                plt.show()
    Returns:
        [type] -- [description]
    """
    def __init__(self,input_array=None,out_channel=None,kernel_size=3,stride=1,pad_width=None):
        self.mu = 0.0
        self.std = kernel_size*kernel_size
        self.conv_2d            = lambda input_img,kernel,k_size,i,j : np.sum(np.multiply(input_img[i:i+k_size,j:j+k_size,:],kernel[:,:,:]))
        self.input_array        = input_array
        self.output             = None
        self.weight             = np.random.normal(self.mu,self.std,(out_channel,kernel_size,kernel_size,input_array.shape[3]))
        self.stride             = stride
        self.pad_width          = pad_width
        self.d_weight           = np.random.normal(self.mu,self.std,(out_channel,kernel_size,kernel_size,input_array.shape[3]))

    def __del__(self):
        del self.conv_2d
        del self.input_array
        del self.output
        del self.weight
        del self.stride
        del self.pad_width
        del self.d_weight


    def forward(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        print('input array size:',self.input_array.shape)
        input_list =[]
        if self.pad_width !=None:
            for j in range(self.input_array.shape[0]):
                for i in range(self.input_array.shape[3]):
                    input_list.append(np.pad(self.input_array[j,:,:,i],self.pad_width))
            self.input_array = np.array(input_list)
            print('modified input array using padding size:',self.input_array.shape)

        i_idx = list(np.linspace(0,self.input_array.shape[1],num=int(self.input_array.shape[1]/self.stride),dtype=np.int))
        j_idx = list(np.linspace(0,self.input_array.shape[2],num=int(self.input_array.shape[2]/self.stride),dtype=np.int))
        output_vol = []
        for inp in range(self.input_array.shape[0]):
            output = []
            for k in range(self.weight.shape[0]):
                output_img = []
                for i in i_idx[:-2]:
                    out_temp = []
                    for j in j_idx[:-2]:
                        out_temp.append(self.conv_2d(self.input_array[inp,:,:,:],self.weight[k,:,:,:],self.weight.shape[1],i,j))
                    output_img.append(out_temp)
                output.append(output_img)
            output_vol.append(output)
        self.output =  np.array(output_vol).swapaxes(1,3).swapaxes(1,2)
        return self.output

class Linear:
    """[Summary]
        x = np.random.rand(1,100)
        ll = Linear(input_array=x,no_output_nodes=3)
        out = ll.forward()
        print('out',out.shape)
    Returns:
        [type] -- [description]


    """
    def __init__(self,input_array=None,no_output_nodes=None):
        self.input_array        = input_array
        self.weight             = np.random.randn(input_array.shape[1],no_output_nodes)
        self.d_weight           = np.random.randn(input_array.shape[1],no_output_nodes)
        self.output             = None

    def __del__(self):
        del self.input_array
        del self.weight
        del self.d_weight
        del self.output

    def forward(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        print('input array size:',self.input_array.shape)
        self.output = np.dot(self.input_array,self.weight)
        return self.output


class Fully_Connected:
    """[Summary]
        x = np.random.rand(1,3,3,3)
        ll = Fully_Connected(input_array=x)
        out = Relu(ll.forward())
        print('out',out.shape)
        plt.imshow(out)
        plt.show()
    Returns:
        [type] -- [description]


    """
    def __init__(self,input_array=None):
        self.input_array        = input_array
        self.output             = None

    def __del__(self):
        del self.input_array
        del self.output

    def forward(self):
        """[summary]

        Returns:
            [type] -- [description]
        """
        self.output =  self.input_array.ravel()
        self.output = np.reshape(self.output,(-1,self.output.shape[0]))
        return self.output


if __name__ == '__main__':
    size = 100
    temp  = np.random.rand(1,size,size,3)
    image = temp
    conv1  = Conv2D(input_array=image,kernel_size=3,out_channel=10,stride=1,pad_width=None)
    relu1 = Relu(conv1.forward())
    conv2  = Conv2D(input_array=relu1,kernel_size=3,out_channel=3,stride=1,pad_width=None)
    relu2 = Relu(conv2.forward())
    fc = Fully_Connected(input_array=relu2)
    out1 = fc.forward()
    ll = Linear(input_array=out1,no_output_nodes=20)
    out2 = ll.forward()
    softmax = Softmax(out2)
    out3 = softmax.forward()

    model = [image,conv1,relu1,conv2,relu2,fc,out1,ll,out2,softmax,out3]
    model.reverse()
    print('input size',image.shape)
    print('output size',model)
    plt.imshow(model[2])
    plt.show()
