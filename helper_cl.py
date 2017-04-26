from CNN_cl import *
import pandas as pd
from collections import OrderedDict

def build_layer(pars):
    pars = dict(pars)
   
    # Convert string into int float  
    for key in pars:
       
        val = pars[key]
        if val.isdigit():
            pars[key] = int(val) 
        elif val.replace('.','',1).isdigit():
            pars[key] = float(val) 
        else:
            pars[key] = val
            
    # Initiate each layer        
    layer = None
    if pars['type'] == "input" :
        layer = input_layer(pars)
        layer.outPlane = np.array(pars['outPlane'],dtype='int32')
        
    if pars['type'] == "conv" :
        layer = conv_layer(pars)
        
    if pars['type'] == "relu" :
        layer = relu_layer()
        
    if pars['type'] == "maxpool" :
        layer = maxpool_layer(pars)
        
    if pars['type'] == "fullyconnected":
        layer = fullyConnected_layer(pars)
        layer.outPlane = np.array(pars['outPlane'],dtype='int32')
        
    if pars['type'] == "softmax" :
        layer = softmax_layer(pars)
        
    if layer:
        layer.bot = pars['bot']
        layer.top = pars['top']
        layer.name = pars['name']
        layer.ltype = pars['type']
       
    else:
        print 'layer type "%s" of layer named %s not found.' %\
            (pars['type'], pars['name'])
    return pars['name'], layer

def init_pars(net, y):
    mu = 0.
    sigma = 1.
    for ly in net.itervalues():
        if ly.bot!='None':
            ly.bot = net[ly.bot]
            ly.inSize = ly.bot.outSize
            
        if ly.top!='None':
            ly.top = net[ly.top]
            
        if ly.ltype == "input":
            pass
        
        if ly.ltype == "conv":
            ly.outSize = np.array( (ly.inSize + 2* ly.padding - ly.filterSize)/ly.stride + 1, dtype='int32')
            ly.inPlane = ly.bot.outPlane
            ly.outPlane = ly.inPlane * ly.nFilter
            ly.weights = np.random.normal(mu, sigma, ly.filterSize * ly.filterSize * ly.outPlane).astype('float32')
            ly.bias = np.random.normal(mu, sigma, ly.outPlane).astype('float32')
            
        if ly.ltype == "relu":
            ly.outSize = ly.bot.outSize
            ly.inPlane = ly.bot.outPlane
            ly.outPlane = ly.inPlane
            
        if ly.ltype == "maxpool":
            ly.outSize = np.array( (ly.inSize + 2 * ly.padding - ly.filterSize)/ly.stride + 1, dtype='int32')
            ly.inPlane = ly.bot.outPlane
            ly.outPlane = ly.inPlane
            
        if ly.ltype == "fullyconnected":
            ly.inSize = ly.bot.outSize
            ly.inPlane = ly.bot.outPlane
            ly.outSize = 1
            nConnects = ly.inPlane * ly.inSize**2* ly.outPlane 
            ly.weights = np.random.normal(mu, sigma, nConnects).astype('float32')  
            ly.bias = np.random.normal(mu,sigma, ly.outPlane).astype('float32')
            
            
        if ly.ltype == "softmax":
            ny = ly.ny
            ly.y = np.array([y]).astype('int32')
            ly.outPlane = np.unique(ny)
           

def forward_pass(net,start):
    start = 0
    for i,l in enumerate(net.itervalues()):
        if l.ltype!='softmax': l.forward()
        if l.ltype=='softmax':
            cost, prob = l.eval()
            return cost
        
def backward_pass(net):
    for l in OrderedDict(reversed(list(net.items()))).itervalues():
        if not (l.ltype=='input' or l.ltype=='softmax'): l.backward()
    

def update_pars(net, eps):
    for l in OrderedDict(reversed(list(net.items()))).itervalues():
        if l.ltype in ['conv','softmax']:
              l.weights = l.weights - eps*l.dw
              l.bias = l.bias -eps*l.db
