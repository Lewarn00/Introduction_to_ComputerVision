import numpy as np
from math import *

'''
    Linear

    Implementation of the linear layer (also called fully connected layer),
    which performs linear transformation on input data: y = xW + b.

    This layer has two learnable parameters:
        weight of shape (input_channel, output_channel)
        bias   of shape (output_channel)
    which are specified and initalized in the init_param() function.

    In this assignment, you need to implement both forward and backward
    computation.

    Arguments:
        input_channel  -- integer, number of input channels
        output_channel -- integer, number of output channels
'''
class Linear(object):

    def __init__(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.init_param()

    def init_param(self):
        self.weight = (np.random.randn(self.input_channel,self.output_channel) * sqrt(2.0/(self.input_channel+self.output_channel))).astype(np.float32)
        self.bias = np.zeros((self.output_channel))

    '''
        Forward computation of linear layer. (3 points)

        Note:  You may want to save some intermediate variables to class
        membership (self.) for reuse in backward computation.

        Arguments:
            input  -- numpy array of shape (N, input_channel)

        Output:
            output -- numpy array of shape (N, output_channel)
    '''
    def forward(self, input):
        ########################
        self.input = input
        N = input.shape[0]
        C = int(np.prod(input.shape)/N)
        output = np.reshape(input, newshape=(N,C)) @ self.weight + self.bias
        ########################
        return output

    '''
        Backward computation of linear layer. (3 points)

        You need to compute the gradient w.r.t input, weight, and bias.
        You need to reuse variables from forward computation to compute the
        backward gradient.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel)

        Output:
            grad_input  -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_weight -- numpy array of shape (input_channel, output_channel), gradient w.r.t weight
            grad_bias   -- numpy array of shape (output_channel), gradient w.r.t bias
    '''
    def backward(self, grad_output):
        ########################
        N = self.input.shape[0]
        rem_shape = self.input.shape
        C = int(np.prod(self.input.shape)/N)
        self.input = np.reshape(self.input, newshape=(N,C))
        grad_input = grad_output @ self.weight.T
        grad_input = np.reshape(grad_input,rem_shape)
        grad_weight = self.input.T @ grad_output
        grad_bias = grad_output.T @ np.ones((N,))
        ########################
        return grad_input, grad_weight, grad_bias

'''
    BatchNorm2d

    Implementation of 2d batch normalization (or BN) layer, which performs
    normalization and rescaling on input data.  Specifically, for input data X
    of shape (N, input_channel, H, W), BN layers first normalize the data along 
    dimensions 0, 2, 3 (with sizes N, H, W), which is done by computing the
    mean mean(X) and variance var(X) across these dimensions, each with a shape
    of (input_channel). Then BN re-scales the normalized data with learnable 
    parameters beta and gamma, both having shape of (input_channel).
    So the forward formula is written as:

        mean(X)[i] = (sum_{n, h, w} X[n,i,h,w]) / (N*H*W)
        var(X)[i] = (sum_{n, h, w} (X[n,i,h,w]-mean(X)[i])^2) / (N*H*W)
        Y = ((X - mean(X)) /  sqrt(Var(x) + eps)) * gamma + beta

    At the same time, BN layer maintains a running_mean and running_variance
    that are updated (with momentum) during forward iteration and would replace
    batch-wise mean(x) and var(x) for testing. The equations are:

        running_mean = (1 - momentum) * mean(x)   +  momentum * running_mean
        running_var =  (1 - momentum) * var(x) +  momentum * running_var

    During test time, since the batch size could be arbitrary, the statistics
    for a batch may not be a good approximation of the data distribution.
    Thus, we instead use running_mean and running_var to perform normalization.
    The forward formula is modified to:

        Y = ((X - running_mean) /  sqrt(running_var + eps)) * gamma + beta

    Overall, BN maintains 4 learnable parameters with shape of (input_channel),
    running_mean, running_var, beta, and gamma.  In this assignment, you need
    to complete the forward and backward computation and handle the cases for
    both training and testing.

    Arguments:
        input_channel -- integer, number of input channel
        momentum      -- float,   the momentum value used for the running_mean and running_var computation
'''
class BatchNorm2d(object):

    def __init__(self, input_channel, momentum = 0.9):
        self.input_channel = input_channel
        self.momentum = momentum
        self.eps = 1e-3
        self.init_param()

    def init_param(self):
        self.r_mean = np.zeros((self.input_channel)).astype(np.float32)
        self.r_var = np.ones((self.input_channel)).astype(np.float32)
        self.beta = np.zeros((self.input_channel)).astype(np.float32)
        self.gamma = (np.random.rand(self.input_channel) * sqrt(2.0/(self.input_channel))).astype(np.float32)

    '''
        Forward computation of batch normalization layer and update of running
        mean and running variance. (3 points)

        You may want to save some intermediate variables to class membership
        (self.) and you should take care of different behaviors during training
        and testing.

        Arguments:
            input -- numpy array (N, input_channel)
            train -- bool, boolean indicator to specify the running mode, True for training and False for testing
    '''
    def forward(self, input, train):
        ########################
        self.input = input
        N, C, H, W = self.input.shape
        x = np.transpose(input, (0,2,3,1)).reshape(-1,C)
        if train == True:
            m = np.mean(x, axis=0) 
            v = np.var(x, axis=0)
            self.v_inv = 1 / np.sqrt(v + self.eps) 
            self.x_m = x - m 
            self.x_norm = self.x_m * self.v_inv 
            output = self.gamma * self.x_norm + self.beta 
            self.r_mean = self.momentum * self.r_mean + (1 - self.momentum) * m
            self.r_var = self.momentum * self.r_var + (1 - self.momentum) * v
        else:
            self.x_norm = (x - self.r_mean) / np.sqrt(self.r_var + self.eps)
            output = self.gamma * x_norm + self.beta
        output = np.transpose(np.reshape(output, (N,H,W,C)), (0,3,1,2))
        ########################
        return output

    '''
        Backward computationg of batch normalization layer. (3 points)
        You need to compute gradient w.r.t input data, gamma, and beta.

        It is recommend to follow the chain rule to first compute the gradient
        w.r.t to intermediate variables, in order to simplify the computation.

        Arguments:
            grad_output -- numpy array of shape (N, input_channel, H, W)

        Output:
            grad_input -- numpy array of shape (N, input_channel, H, W), gradient w.r.t input
            grad_gamma -- numpy array of shape (input_channel), gradient w.r.t gamma
            grad_beta  -- numpy array of shape (input_channel), gradient w.r.t beta
    '''
    def backward(self, grad_output):
        ########################
        N1, C, H, W = grad_output.shape
        out = np.transpose(grad_output, (0,2,3,1)).reshape(-1,C)
        N2 = self.x_m.shape[0]
        grad_gamma = np.sum(out * self.x_norm, axis=0)
        grad_beta = np.sum(out, axis=0)
        gradv = np.sum(out * self.gamma * self.x_m, axis=0) * -0.5 * (self.v_inv ** 3)
        gradm = np.sum(out * self.gamma * -self.v_inv, axis=0) 
        gradm = gradm + gradv * -2 * np.mean(self.x_m, axis=0)
        grad_input = out * self.gamma * self.v_inv + gradv * (2 / N2) * self.x_m + gradm * (1 / N2)
        grad_input = np.reshape(grad_input, (N1, C, H, W))
        ########################
        return grad_input, grad_gamma, grad_beta

'''
    ELU

    Implementation of ELU (exponential linear unit) layer.  ELU is the
    non-linear activation function that applies an exponential decay to
    all negative values.
    The formula is: y = x if x > 0, exp(x)-1 otherwise.

    This layer has no learnable parameters and you need to implement both
    forward and backward computation.

    Arguments:
        None
'''
class ELU(object):
    def __init__(self):
        pass

    '''
        Forward computation of ELU. (3 points)

        You may want to save some intermediate variables to class membership
        (self.)

        Arguments:
            input  -- numpy array of arbitrary shape

        Output:
            output -- numpy array having the same shape as input.
    '''
    def forward(self, input):
        ########################
        def exponential(x):
            result = 0
            if x > 0:
                result = x
            else:
                result = np.exp(x)-1
            return result

        self.input = input
        output = np.empty_like(input)
        output[:] = input
        vectorized = np.vectorize(exponential)
        output = np.array(list(map(vectorized, output)))
        #output = np.where(input > 0, input, np.exp(input)-1)
        ########################
        return output

    '''
        Backward computation of ELU. (3 points)

        You can either modify grad_output in-place or create a copy.

        Arguments:
            grad_output -- numpy array having the same shape as input

        Output:
            grad_input  -- numpy array has the same shape as grad_output. gradient w.r.t input
    '''
    def backward(self, grad_output):
        ########################
        grad_output[self.input<0] *= np.exp(self.input[self.input<0])
        grad_input = grad_output
        ########################
        return grad_input

'''
    CrossEntropyLossWithSoftmax

    Implementation of the combination of softmax function and cross entropy
    loss.  In classification tasks, we usually first apply the softmax function
    to map class-wise prediciton scores into a probability distribution over
    classes.  Then we use the cross entropy loss to maximize the likelihood of
    the ground truth class' prediction.  Since softmax includes an exponential
    term and cross entropy includes a log term, we can simplify the formula by
    combining these two functions together, so that log and exp operations
    cancel out.  This way, we also avoid some precision loss due to floating
    point numerical computation.

    If we ignore the index on batch size and assume there is only one ground
    truth per sample, the formula for softmax and cross entropy loss are:

        Softmax: prob[i] = exp(x[i]) / \sum_{j}exp(x[j])
        Cross_entropy_loss:  - 1 * log(prob[gt_class])

    Combining these two functions togther, we have:

        cross_entropy_with_softmax: -x[gt_class] + log(\sum_{j}exp(x[j]))

    In this assignment, you will implement both forward and backward
    computation.

    Arguments:
        None
'''
class CrossEntropyLossWithSoftmax(object):
    def __init__(self):
        pass

    '''
        Forward computation of cross entropy with softmax. (3 points)

        You may want to save some intermediate variables to class membership
        (self.)

        Arguments:
            input    -- numpy array of shape (N, C), the prediction for each class, where C is number of classes
            gt_label -- numpy array of shape (N), it is an integer array and the value range from 0 to C-1 which
                        specify the ground truth class for each input
        Output:
            output   -- numpy array of shape (N), containing the cross entropy loss on each input
    '''
    def forward(self, input, gt_label):
        ########################
        self.input = input
        self.labels = gt_label
        self.sumx = np.sum(np.exp(input), axis=1)
        temp_output = [-x[gt_label[i]] + np.log(self.sumx[i]) for i,x in enumerate(input)]
        output = np.array(temp_output).astype(np.float32)
        ########################
        return output

    '''
        Backward computation of cross entropy with softmax. (3 points)

        It is recommended to resue the variable(s) in forward computation
        in order to simplify the formula.

        Arguments:
            grad_output -- numpy array of shape (N)

        Output:
            output   -- numpy array of shape (N, C), the gradient w.r.t input of forward function
    '''
    def backward(self, grad_output):
        ########################
        N, C  = self.input.shape
        deriv = np.zeros((N,N,C))
        for i in range(N):
            deriv[i,0] = deriv[i,0] + np.exp(self.input[i])/self.sumx[i]
            deriv[i,0,self.labels[i]] = deriv[i,0,self.labels[i]] - 1
        grad_input = grad_output @ deriv
        ########################
        return grad_input

'''
    im2col (3 points)

    Consider 4 dimensional input tensor with shape (N, C, H, W), where:
        N is the batch dimension,
        C is the channel dimension, and
        H, W are the spatial dimensions.

    The im2col functions flattens each slidding kernel-sized block
    (C * kernel_h * kernel_w) on each sptial location, so that the output has
    the shape of (N, (C * kernel_h * kernel_w), out_H, out_W) and we can thus
    formuate the convolutional operation as matrix multiplication.

    The formula to compute out_H and out_W is the same as to compute the output
    spatial size of a convolutional layer.

    Arguments:
        input_data  -- numpy array of shape (N, C, H, W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- numpy array of shape (N, (C * kernel_h * kernel_w), out_H, out_W)
'''
def im2col(input_data, kernel_h, kernel_w, stride, padding):
    ########################
    padded = np.pad(input_data, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
    N, C, H, W = input_data.shape
    output_height = ((H + 2 * padding - kernel_h) // stride) + 1
    output_width = ((W + 2 * padding - kernel_w) // stride) + 1
    '''
    output = np.zeros((N, C, kernel_h,kernel_w,output_height,output_width))
    for row in range(kernel_h):
    	max_row = row + stride*output_height
    	for col in range(kernel_w):
    		max_col = col + stride*output_width
    		output[:,:,row,col,:,:] = padded[:,:,row:max_row:stride,col:max_col:stride]
    output_data = output.transpose(0,4,5,1,2,3).reshape(N*output_height*output_width,-1)
    '''
    startl = np.tile(np.repeat(np.arange(kernel_h), kernel_w), C)
    endl = stride * np.repeat(np.arange(output_height), output_width)
    x = startl.reshape(-1, 1) + endl.reshape(1, -1)
    starts = np.tile(np.tile(np.arange(kernel_w), kernel_h), C)
    ends = stride * np.tile(np.arange(output_width), output_height)
    y = starts.reshape(-1, 1) + ends.reshape(1, -1)
    z = np.repeat(np.arange(C), kernel_h * kernel_w).reshape(-1, 1)
    output_data = np.concatenate(padded[:, z, x, y], axis=-1)
    ########################
    return output_data
'''
    col2im (3 points)

    Consider a 4 dimensional input tensor with shape:
        (N, (C * kernel_h * kernel_w), out_H, out_W)
    where:
        N is the batch dimension,
        C is the channel dimension,
        out_H, out_W are the spatial dimensions, and
        kernel_h and kernel_w are the specified kernel spatial dimension.

    The col2im function calculates each combined value in the resulting array
    by summing all values from the corresponding sliding kernel-sized block.
    With the same parameters, the output should have the same shape as
    input_data of im2col.  This function serves as an inverse subroutine of
    im2col, so that we can formuate the backward computation in convolutional
    layers as matrix multiplication.

    Arguments:
        input_data  -- numpy array of shape (N, (C * kernel_H * kernel_W), out_H, out_W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- output_array with shape (N, C, H, W)
'''
def col2im(input_data, kernel_h, kernel_w, stride=1, padding=0, shape = None):
    ########################
    N, C, H, W = shape
    h_padding = H + 2 * padding
    w_padding = W + 2 * padding
    padded = np.zeros((N, C, h_padding, w_padding), dtype=input_data.dtype)
    output_height = ((H + 2 * padding - kernel_h) // stride) + 1
    output_width = ((W + 2 * padding - kernel_w) // stride) + 1    
    startl = np.tile(np.repeat(np.arange(kernel_h), kernel_w), C)
    endl = stride * np.repeat(np.arange(output_height), output_width)
    x = startl.reshape(-1, 1) + endl.reshape(1, -1)
    starts = np.tile(np.tile(np.arange(kernel_w), kernel_h), C)
    ends = stride * np.tile(np.arange(output_width), output_height)
    y = starts.reshape(-1, 1) + ends.reshape(1, -1)
    z = np.repeat(np.arange(C), kernel_h * kernel_w).reshape(-1, 1)
    reshaped = input_data.reshape(C * kernel_h * kernel_w, -1, N).transpose(2, 0, 1)
    np.add.at(padded, (slice(None), z, x, y), reshaped)
    if padding == 0:
        output_data = padded
    else:
        output_data = padded[:, :, padding:-padding, padding:-padding]
    ########################
    return output_data
'''
    Conv2d

    Implementation of convolutional layer.  This layer performs convolution
    between each sliding kernel-sized block and convolutional kernel.  Unlike
    the convolution you implemented in HW1, where you needed flip the kernel,
    here the convolution operation can be simplified as cross-correlation (no
    need to flip the kernel).

    This layer has 2 learnable parameters, weight (convolutional kernel) and
    bias, which are specified and initalized in the init_param() function.
    You need to complete both forward and backward functions of the class.
    For backward, you need to compute the gradient w.r.t input, weight, and
    bias.  The input arguments: kernel_size, padding, and stride jointly
    determine the output shape by the following formula:

        out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

    You need to use im2col, col2im inside forward and backward respectively,
    which formulates the sliding window computation in a convolutional layer as
    matrix multiplication.

    Arguments:
        input_channel  -- integer, number of input channel which should be the same as channel numbers of filter or input array
        output_channel -- integer, number of output channel produced by convolution or the number of filters
        kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                          width of kernel size.
        padding        -- zero padding added on both sides of input array
        stride         -- integer, stride of convolution.
'''
class Conv2d(object):
    def __init__(self, input_channel, output_channel, kernel_size, padding = 0, stride = 1):
        self.output_channel = output_channel
        self.input_channel = input_channel
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride
        self.init_param()

    def init_param(self):
        self.weight = (np.random.randn(self.output_channel, self.input_channel, self.kernel_h, self.kernel_w) * sqrt(2.0/(self.input_channel + self.output_channel))).astype(np.float32)
        self.bias = np.zeros(self.output_channel).astype(np.float32)

    '''
        Forward computation of convolutional layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, output_chanel, out_H, out_W)
    '''
    def forward(self, input):
        ########################
        self.input = input
        N, C, H, W = input.shape
        C = self.output_channel
        output_height = ((H + 2 * self.padding - self.kernel_h) // self.stride) + 1
        output_width = ((W + 2 * self.padding - self.kernel_w) // self.stride) + 1  
        self.col = im2col(input, self.kernel_h, self.kernel_w, self.stride, self.padding)
        w = self.weight.reshape((self.output_channel, -1))
        b = self.bias.reshape(-1, 1)
        output = w @ self.col + b
        output = np.array(np.hsplit(output, N)).reshape((N, C, output_height, output_width))
        ########################
        return output
    '''
        Backward computation of convolutional layer. (3 points)

        You need col2im and saved variables from forward() in this function.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
            grad_weight -- numpy array of shape(output_channel, input_channel, kernel_h, kernel_w), gradient w.r.t weight
            grad_bias   -- numpy array of shape(output_channel), gradient w.r.t bias
    '''

    def backward(self, grad_output):
        ########################
        '''
        N, C, H, W  = self.input.shape
        grad_bias = np.sum(grad_output, axis=(0,2,3))
        X,Y,Z,D = grad_output.shape
        grad_output = np.array(np.vsplit(grad_output.reshape(X * Y, Z * D), N))
        grad_output = np.concatenate(grad_output, axis=-1)
        find_grad = self.weight.T @ grad_output
        grad_weight = grad_output @ self.col.T
        grad_input = col2im(find_grad, self.kernel_h, self.kernel_w, self.stride, self.padding, self.input.shape)
        grad_weight = grad_weight.reshape((grad_weight.shape[0], self.input_channel, self.kernel_h, self.kernel_w))
        '''
        grad_bias = np.sum(grad_output, axis=(0,2,3))
        grad_bias = grad_bias.reshape(self.output_channel,-1)
        grad_output_r = grad_output.transpose(1,2,3,0).reshape(self.output_channel,-1)
        grad_weight = grad_output_r @ self.col.T
        grad_weight = grad_weight.reshape(np.shape(self.weight))
        weight_r = self.weight.reshape(self.output_channel,-1)
        find_grad = weight_r.T @ grad_output_r
        grad_input = col2im(find_grad, self.kernel_h, self.kernel_w, self.stride, self.padding, self.input.shape)
        ########################
        return grad_input, grad_weight, grad_bias

'''
    AvgPool2d

    Implementation of average pooling layer.  For each sliding kernel-sized block,
    avgpool2d computes the spatial average along each channel.  This layer has
    no learnable parameters.

    You need to complete both forward and backward functions of the layer.
    For backward, you need to compute the gradient w.r.t input.  Similar as
    conv2d, the input argument, kernel_size, padding and stride jointly
    determine the output shape by the following formula:

        out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

    You may use im2col, col2im inside forward and backward, respectively.

    Arguments:
        kernel_size    -- integer or tuple, spatial size of convolution kernel. 
        If it's tuple, it specifies the height and width of kernel size.
        padding        -- zero padding added on both sides of input array
        stride         -- integer, stride of convolution.
'''
class AvgPool2d(object):
    def __init__(self, kernel_size, padding = 0, stride = 1):
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride

    '''
        Forward computation of avg pooling layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, input_channel, out_H, out_W)
    '''
    def forward(self, input):
        ########################
        self.input = input
        N,C, H, W = input.shape
        output_height = ((H + 2 * self.padding - self.kernel_h) // self.stride) + 1
        output_width = ((W + 2 * self.padding - self.kernel_w) // self.stride) + 1  
        col = im2col(input, self.kernel_h, self.kernel_w, self.stride, self.padding)
        col = col.reshape(C,col.shape[0]//C,-1)
        pool = np.array(np.hsplit(np.mean(col, axis=1), N))
        output = pool.reshape(N, C, output_height, output_width)
        ########################
        return output
        '''
        #AvgPool2d forward without im2col, please uncomment and use if other version breaks
        self.input = input
        N, C, H, W = input.shape
        out_height = (H - self.kernel_h + 2 * self.padding) // self.stride + 1
        out_width = (W - self.kernel_w + 2 * self.padding) // self.stride + 1
        output = np.zeros(shape=(N,C,out_height,out_width))
        for n in range(N):
          for c in range(C):
            for h in range(out_height):
              start_h = self.stride * h
              end_h = self.stride * h + self.kernel_h
              for w in range(out_width):
                start_w = self.stride * w
                end_w = self.stride * w + self.kernel_w
                x_pool = input[n, c, start_h:end_h, start_w:end_w]
                output[n, c, h, w]  = np.mean(x_pool)
        return output
        '''
        
    '''
        Backward computation of avg pooling layer. (3 points)

        You should use col2im and saved variable(s) from forward().

        Arguments:
            grad_output -- numpy array of shape (N, input_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
    '''
    def backward(self, grad_output):
        ########################
        N, C, H, W = self.input.shape
        output_height = ((H + 2 * self.padding - self.kernel_h) // self.stride) + 1
        output_width = ((W + 2 * self.padding - self.kernel_w) // self.stride) + 1  
        flatten = grad_output.reshape(C, -1) / (self.kernel_h * self.kernel_w)
        col = np.repeat(flatten, self.kernel_h * self.kernel_w, axis=0)
        ncol = col2im(col, self.kernel_h, self.kernel_w, self.stride, self.padding, self.input.shape)
        ncol = np.array(np.hsplit(ncol.reshape(N, -1), C))
        grad_input = ncol.reshape(N, C, H, W)
        ########################
        return grad_input
        '''
        #AvgPool2d backward without col2im, please uncomment and use if other version breaks
        def distribute(dz, shape):
            (C2, output_width) = shape
            avg = dz / (C2 * output_width)
            d = np.ones(shape) * avg
            return d
        N, C, H, W = self.input.shape
        N2, C2, output_height, output_width = grad_output.shape
        grad_input = np.zeros(shape=self.input.shape)
        for n in range(N2):
          for c in range(C2):
            for h in range(output_height):
              start_h = self.stride * h
              end_h = self.stride * h + self.kernel_h
              for w in range(output_width):
                start_w = self.stride * w
                end_w = self.stride * w + self.kernel_w
                temp = grad_output[n, c, h, w]
                shape = (self.kernel_h, self.kernel_w)
                grad_input[n, c, start_h:end_h, start_w:end_w] += distribute(temp, shape)
        return grad_input
        '''
        
        
