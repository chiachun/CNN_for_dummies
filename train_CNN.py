from CNN2 import *
from collections import OrderedDict
from ConfigParser import SafeConfigParser
import pandas as pd
import CNN2
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

def calc_wtop(net, N, ny):
    
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
            ly.ny = ny
            ly.weights = np.random.normal(mu, sigma, ly.botl.wtop**2 * ny)  
            ly.weights = ly.weights.reshape(ly.botl.wtop**2, ny)
            ly.bias = np.random.normal(mu,sigma,ny)
            ly.wtop = ly.ny
        
# layer names in the config file
layernames = ['input','conv1','relu1','maxpool1','softmax']
parser = SafeConfigParser()
parser.optionxform = str  # make option names case sensitive
parser.read('config.ini')

# Read parameters from the config file
net = OrderedDict()
for l in layernames:
    pars = parser.items(l)
    name, layer = build_layer(pars)
    if layer:
        net[name] = layer


df = pd.read_csv('train_small.csv',index_col=0)
y = np.array(df.iloc[:,0])
data = np.array(df.iloc[:,1:]).reshape(-1,28,28)
N = data.shape[0]
ny = np.unique(y).size
calc_wtop(net, N, ny)
net['input'].top = data

def forward_propa(net,start):
    start = 0
    for i,l in enumerate(net.itervalues()):
        if i < start: continue
        if l.ltype!='softmax': l.forward()
        if l.ltype=='softmax':
            cost, prob = l.eval(y)
            return cost
def backward_propa(net):
    for l in OrderedDict(reversed(list(net.items()))).itervalues():
        if not (l.ltype=='input' or l.ltype=='softmax'): l.backward()
    

def update(net, eps):
    for l in OrderedDict(reversed(list(net.items()))).itervalues():
        if l.ltype in ['conv','softmax']:
              l.weights = l.weights - eps*l.dw
              l.bias = l.bias -eps*l.db

for i in range(200):
    cost = forward_propa(net,0)
    print "%dth iteration: cost = %f" % (i, cost) 
    backward_propa(net)
    update(net,0.001)




