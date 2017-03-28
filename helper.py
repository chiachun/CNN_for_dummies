from CNN import *
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
        
    if pars['type'] == "conv" :
        layer = conv_layer(pars)
        
    if pars['type'] == "relu" :
        layer = relu_layer()
        
    if pars['type'] == "maxpool" :
        layer = maxpool_layer(pars)
        
    if pars['type'] == "softmax" :
        layer = softmax_layer(pars)
        
    if layer:
        layer.botl = pars['bot']
        layer.topl = pars['top']
        layer.name = pars['name']
        layer.ltype = pars['type']
    else:
        print 'layer type "%s" of layer named %s not found.' %\
            (pars['type'], pars['name'])
    return pars['name'], layer

def calc_wtop(net, N, y):
    
    for ly in net.itervalues():
        ly.N = N
        if ly.botl!='None':
            ly.botl = net[ly.botl]
        if ly.topl!='None':
            ly.topl = net[ly.topl]
        if ly.ltype == "input":
            pass
        if ly.ltype == "conv":
            ly.wtop = (ly.botl.wtop - ly.size + 2*ly.pad)/ly.stride + 1
        if ly.ltype == "relu":
            ly.wtop = ly.botl.wtop
        if ly.ltype == "maxpool":
            ly.wtop = (ly.botl.wtop - ly.stride)/ly.stride + 1
        if ly.ltype == "softmax":
            ny = ly.ny
            ly.weights = np.random.normal(mu, sigma, ly.botl.wtop**2 * ny)  
            ly.weights = ly.weights.reshape(ly.botl.wtop**2, ny)
            ly.bias = np.random.normal(mu,sigma,ny)
            ly.y = y
            ly.wtop = np.unique(ny)
           

def forward_pass(net,start):
    start = 0
    for i,l in enumerate(net.itervalues()):
        if i < start: continue
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
