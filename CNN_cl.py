import random
import numpy as np
from forward_cl import forward_conv, forward_conv_quick, forward_maxpool
from forward_cl import forward_fc_direct, forward_fc_block
import time


def stretch(x, h, s):
    N, n1, n2 = x.shape
    out = []
    for k in range(N):
        for i in range(0,n1-h+1,s):
            for j in range(0,n2-h+1,s):
                mat = x[k,i:i+h,j:j+h].flatten().tolist()
                out.extend(mat)
    return np.array(out).reshape(N * ((n1-h)/s + 1) * ((n2-h)/s + 1), h*h )



class softmax_layer:
    def __init__(self,pars):
        self.score = 0
        self.N = 0
        self.ny = pars['ny']


class fullyConnected_layer:
    def __init__(self,pars):
        self.N = 0
        self.dw = None
        self.db = None
    
    def forward_cl(self,x):
        pars = (self.N, self.inPlane, self.outPlane, self.inSize)
        out = forward_fc_block(x, pars, self.weights, self.bias)
        return out
    # This function uses opencl program to calculate the inner product
    # directly. (No parallelism).
    def forward_cl1(self,x):
        pars = (self.N, self.inPlane, self.outPlane, self.inSize)
        out = forward_fc_direct(x, pars, self.weights, self.bias)
        return out
    
    def forward(self,x):
        x = x.reshape(self.N[0],-1) 
        w = self.weights.reshape(-1,self.outPlane)
        return np.dot(x,w)
    
class maxpool_layer:
    def __init__(self, pars):
        self.N = 0
        self.filterSize = np.array([pars['size']], dtype='int32')
        self.stride = np.array([pars['stride']], dtype='int32') 
        self.padding = np.array([pars['padding']], dtype='int32')
        self.inSize = 0
        self.outSize = 0
      
        
    def forward_cl(self,x):
        pars = (self.N, self.inPlane, 1, self.filterSize,
                self.stride, self.padding, self.inSize, self.outSize)
        output, sel = forward_maxpool(x, pars)
        self.weights = sel
        return output
    
    def forward(self,x):
        x = x.reshape(self.N[0], self.bot.outPlane[0], self.inSize[0], self.inSize[0])
        fsiz = self.filterSize[0]; pad = self.padding[0]; stride = self.stride[0]
        osiz = self.outSize[0]
        out_pad = np.zeros(self.N[0] * self.outPlane[0] * osiz * osiz).reshape(self.N[0], self.outPlane[0], osiz, osiz)
        padinSize = self.inSize[0] + 2
        weight_pad = np.zeros(self.N[0]*self.bot.outPlane[0]*padinSize*padinSize).reshape(self.N[0],self.bot.outPlane[0], padinSize, padinSize)
        for n in range(self.N):
            for p in range(self.inPlane[0]):
                x_pad = np.pad(x[n,p,:,:], ((pad,pad),(pad,pad)), 'constant')
                for iy in range(0, osiz):
                    for ix in range(0, osiz):
                        h1 = iy * stride
                        h2 = iy * stride + fsiz
                        w1 = ix * stride
                        w2 = ix * stride + fsiz
                        pool = x_pad[ h1:h2, w1:w2]
                        out_pad[n, p, iy,ix] = np.amax(pool)
                        idx = np.argmax(pool)
                        weight_pad[n, p, h1 + idx/fsiz, w1 + idx%fsiz] = 1
        self.weights = weight_pad
        return out_pad

class relu_layer:
    def __init__(self):
        pass
    
    def forward(self,x):
        out = np.maximum(0,x)
        self.da = out.copy()
        self.da[self.da.nonzero()] = 1
        return out
    
    def backward(self,delta):
        return np.multiply(self.da,delta)

class input_layer:
    def __init__(self, pars):
        self.outSize = np.array([pars['width']], dtype='int32')
        self.top = ''
    
    def forward(self):
        pass
    
class conv_layer:
    def __init__(self, pars):
        self.N = 0
        self.inplane = None
        self.nFilter = np.array([pars['depth']], dtype='int32')
        self.filterSize = np.array([pars['size']], dtype='int32')
        self.stride = np.array([pars['stride']], dtype='int32') 
        self.padding = np.array([pars['padding']], dtype='int32')
        self.inSize = 0
        self.outSize = 0
        self.weights = None
        self.bias = None
    # This function deals with padding by turning off filter pixels
    # at padded pixels.
    def forward_cl(self,x):
        pars = (self.N, self.inPlane, self.nFilter, self.filterSize,
                self.stride, self.padding, self.inSize, self.outSize)
        output = forward_conv_quick(x, pars, self.weights, self.bias)
        return output
    # This function pads input with zeros before sending data into opencl program.
    # This function is computationally costy since padding traverses through the whole
    # array to add zeros in.
    def forward_cl1(self,x):
        pad = self.padding[0]
        x = x.reshape(self.N[0], self.bot.outSize[0], self.bot.outSize[0])
        padinSize = self.inSize[0] + 2*pad
        self.x = np.zeros(self.N[0]* padinSize * padinSize).reshape(self.N[0], padinSize, padinSize)
        for n in range(self.N[0]):
            self.x[n,:,:] = np.pad(x[n,:,:], ((pad,pad),(pad,pad)), 'constant')
        pars = (self.N, self.inPlane, self.nFilter, self.filterSize,
                self.stride, self.padding, padinSize, self.outSize)
        output = forward_conv(self.x.astype('float32').flatten(), pars, self.weights, self.bias)
        return output
    
    def forward(self,x):
        pad = self.padding[0]
        x = x.reshape(self.N[0], self.bot.outSize[0], self.bot.outSize[0])
        padinSize = self.inSize[0] + 2*pad
        self.x2 = np.empty((self.N[0], padinSize, padinSize))
        for n in range(self.N[0]):
            self.x2[n,:,:] = np.pad(x[n,:,:], ((pad,pad),(pad,pad)), 'constant')
        self.x_str = stretch(self.x2, h=self.filterSize[0], s=self.stride[0])
        
        # stretch the filter
        self.ws = self.weights
        bs = self.bias
    
        self.ws = self.ws.reshape(self.nFilter[0],self.filterSize[0] * self.filterSize[0])
        self.out = np.empty((self.nFilter[0], self.N[0] * self.outSize[0] * self.outSize[0]))
        
        for F in range(self.nFilter[0]): 
            self.out[F,:] = np.dot(self.x_str,self.ws[F]) + self.bias[F]
        self.out = np.swapaxes(self.out.reshape(self.nFilter[0],self.N[0],-1), 0,1)
        return self.out.flatten()
