import random
import numpy as np


mu = 0
sigma = 0.01

# stride = s, filter = h*h. assuming padding = 0 for simplicy
def stretch(x, h, s):
    N, n1, n2 = x.shape
    out = []
    for k in range(N):
        for i in range(0,n1-h+1,s):
            for j in range(0,n2-h+1,s):
                mat = x[k,i:i+h,j:j+h].flatten().tolist()
                out.extend(mat)
    return np.array(out).reshape(N * ((n1-h)/s + 1) * ((n2-h)/s + 1), h*h )


           
# this is a virtual class
class layer:
    def __init__(self):
        pass
    
    def forward(self):
        pass

    def backward(self):
        pass
    
class input_layer(layer):
    def __init__(self, pars):
        self.wtop = pars['width']
        self.top = None
    def forward(self):
        pass
    
class conv_layer(layer):
    def __init__(self, pars):
        self.size = pars['size'] # size of the filter
        self.stride = pars['stride'] 
        self.pad = pars['padding']
        self.depth = pars['depth']
        self.top = None
        self.bot = None
        self.wtop = 0
        self.N = 0
        self.weights = np.random.normal(mu, sigma, self.size * self.size)  
        self.weights = self.weights.reshape(self.size, self.size)
        self.bias = np.random.normal(mu, sigma, 1)
        self.botl = None
        self.topl = None
        
    def forward(self):
        self.bot = self.botl.top
        bot = self.bot
        ws = self.weights
        bs = self.bias
        
        # Perform convolution at (11-3)+1=9 locations on both height and width 
        # by stretching the input image and convolution filter into a large matrix
        # for easy dot operation.
        bot_str = stretch(bot, h=self.size, s=self.stride)
        #bots.shape = (n, 3*3,9*9) = (9 pixels, 81 locations)

        # stretch the filter
        ws = np.array(ws.flatten().tolist())
        ws = ws.reshape(self.size * self.size)
        top = np.dot(bot_str,ws) + self.bias # out.shape = (1*81)
        self.top = top.reshape(self.N, self.wtop, self.wtop)

    def backward(self):
        dtop = self.topl.dbot
        ws = self.weights
        bs = self.bias
        pad = self.pad
        bot = self.bot
        (__, ny, nx) = dtop.shape
        
        dbot = np.zeros_like(bot).astype('float')
        dw = np.zeros_like(ws)
        db = np.zeros_like(bs)
        self.dbot = dbot
        for n in xrange(self.N):
            dbot_pad = np.pad(dbot[n,:,:], ((pad,pad),(pad,pad)), 'constant')
            bot_pad = np.pad(bot[n,:,:], ((pad,pad),(pad,pad)), 'constant')
            for iy in xrange(ny):
                for ix in xrange(nx):
                    h1 = iy * self.stride
                    h2 = iy * self.stride + self.size
                    w1 = ix * self.stride
                    w2 = ix * self.stride + self.size
                    dbot_pad[h1:h2, w1:w2] += ws[:,:] * dtop[n, iy, ix]
                    dw[:,:] += bot_pad[h1:h2, w1:w2] * dtop[n, iy, ix]
                    db += dtop[n,iy,ix]
            y0,x0 = dbot[0].shape
            dbot[n,:,:] = dbot_pad[:y0-pad, :x0-pad]
        self.dw = dw/self.N
        self.db = db/self.N


# ReLu layer has no weight/bias
# Only dbot has to be calculated in backprop

class relu_layer:
    def __init__(self):
        self.top = None
        self.wtop = 0
        
    def forward(self):
        self.bot = self.botl.top
        bot = self.bot
        self.top = np.maximum(0,bot)
        self.da = self.top.copy().flatten()
        self.da[self.da.nonzero()] = 1
        self.da = self.da.reshape(self.top.shape)
        
    def backward(self):
        dtop = self.topl.dbot
        self.dbot = np.multiply(self.da,dtop)

# Max pool layer has no weight/bias
# Only dbot has to be calculated in backprop
class maxpool_layer:
    def __init__(self, pars):
        self.size = pars['size']
        self.stride = pars['stride']
        self.pad = pars['padding']
        self.top = None
        self.wtop = None
        self.N = 0
        self.dbot = None
        
    def forward(self):
        self.bot = self.botl.top
        bot = self.bot
        N, ny, nx = bot.shape
        h = self.size; s = self.stride; pad = self.pad
        ny = ny - h + pad + 1
        nx = nx - h + pad + 1
        top_pad = np.zeros(self.N*self.wtop*self.wtop).reshape(self.N,self.wtop,self.wtop)
        weight_pad = np.zeros_like(bot)
        for n in range(N):
            bot_pad = np.pad(bot[n,:,:], ((pad,pad),(pad,pad)), 'constant')
            for iy in range(0, self.wtop):
                for ix in range(0, self.wtop):
                    h1 = iy * self.stride
                    h2 = iy * self.stride + self.size
                    w1 = ix * self.stride
                    w2 = ix * self.stride + self.size
                    pool = bot_pad[ h1:h2, w1:w2]
                    top_pad[n, iy,ix] = np.amax(pool)
                    idx = np.argmax(pool)
                    weight_pad[n, h1+idx/h, w1+idx%h] = 1
        self.top = top_pad
        self.weights = weight_pad

    def backward(self):
        top = self.topl
        dtop = top.dbot.reshape(self.N, self.wtop, self.wtop)
        pad = self.pad
        bot = self.bot   
        dbot = np.zeros_like(bot).astype('float')
   
        
        for n in xrange(self.N):
            dbot_pad = np.pad(dbot[n,:,:], ((pad,pad),(pad,pad)), 'constant')
            for iy in xrange(self.wtop):
                for ix in xrange(self.wtop):
                    h1 = iy * self.stride
                    h2 = iy * self.stride + self.size
                    w1 = ix * self.stride
                    w2 = ix * self.stride + self.size
                    dbot_pad[h1:h2, w1:w2] += self.weights[n,h1:h2,w1:w2] * dtop[n, iy, ix]
            y0,x0 = dbot[0].shape
            dbot[n,:,:] = dbot_pad[:y0-pad, :x0-pad]
        self.dbot = dbot
        
class softmax_layer:
    def __init__(self,pars):
        self.weights = None
        self.bias = None
        self.top = 0
        self.N = 0
        self.dbot = None
        self.dw = None
        self.db = None
        self.ny = pars['ny']
      
        
    def eval(self):
        y = self.y
        # stretch bot
        self.bot = self.botl.top
        bot = self.bot
        N = bot.shape[0]
        D = np.prod(bot.shape[1:])
        bot = np.reshape(bot, (N, D)) ; self.bot = bot
        z = np.dot(bot,self.weights) + self.bias 
        
        # e(x)->e(x-m) subtract by the maximum to get always negative input
        # toavoid an exploding e(x)

        prob = np.exp(z - np.max(z, axis=1, keepdims=True))
        prob = prob / np.sum(prob, axis=1, keepdims=True)
        # cost
        self.top = - np.sum(np.log(prob[np.arange(N), y])) / N # avg log likelihood
        self.delta = prob.copy()
        
        # delta_L
        self.delta[np.arange(N), y] = self.delta[np.arange(N),y]- 1.
        self.dw = np.dot(bot.T, self.delta) /self.N
        self.db = np.sum(self.delta, axis=0)/self.N
    
        # weight * delta_L
        self.dbot = np.dot(self.delta,self.weights.T).reshape(N,self.botl.wtop, self.botl.wtop)
        
        return self.top, prob
    
    def predict(self):
        self.bot = self.botl.top
        bot = self.bot
        N = bot.shape[0]
        D = np.prod(bot.shape[1:])
        bot = np.reshape(bot, (N, D)) ; self.bot = bot
        z = np.dot(bot,self.weights) + self.bias 
        
        # e(x)->e(x-m) subtract by the maximum to get always negative input
        # toavoid an exploding e(x)

        prob = np.exp(z - np.max(z, axis=1, keepdims=True))
        prob = prob / np.sum(prob, axis=1, keepdims=True)

        return prob
