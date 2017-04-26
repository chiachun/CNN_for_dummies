from collections import OrderedDict
from ConfigParser import SafeConfigParser
from helper_cl import forward_pass, backward_pass, update_pars, build_layer, init_pars
import pandas as pd
import numpy as np
import time

# layer names in the config file
layernames = ['input','conv1','relu1','maxpool1','fc1','softmax']
parser = SafeConfigParser()
parser.optionxform = str  # make option names case sensitive
parser.read('configs/config_t.ini')

# Read parameters from the config file
net = OrderedDict()
for l in layernames:
    pars = parser.items(l)
    name, layer = build_layer(pars)
    if layer:
        net[name] = layer


df = pd.read_csv('train.csv')
data2 = np.array(df.iloc[:1000,1:],dtype='float32')
N = data2.shape[0]
init_pars(net, [1])
for ly in net.itervalues():
    ly.N = np.array([N])


net['input'].data = data2.flatten()
ts = time.time()
out1_0 = net['conv1'].forward_cl(net['input'].data)
out2_0 = net['relu1'].forward(out1_0)
out3_0 = net['maxpool1'].forward_cl(out2_0)
out4_0 = net['fc1'].forward_cl(out3_0)
print "forward_quick: t=  ", time.time() - ts

import matplotlib.pyplot as plt


ts = time.time()
out1 = net['conv1'].forward(net['input'].data)
out2 = net['relu1'].forward(out1)
out3 = net['maxpool1'].forward(out2)
out4 = net['fc1'].forward(out3)
print "forward: t=  ", time.time() - ts

print "Difference fraction: %f " %  (sum( (out4_0-out4.flatten()) / out4_0 ) / sum(out4_0))
