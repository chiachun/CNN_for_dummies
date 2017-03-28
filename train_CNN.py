from collections import OrderedDict
from ConfigParser import SafeConfigParser
from helper import forward_pass, backward_pass, update_pars, build_layer, calc_wtop
import pandas as pd
import numpy as np
# layer names in the config file
layernames = ['input','conv1','relu1','maxpool1','softmax']
parser = SafeConfigParser()
parser.read('config.ini')

# Read parameters from the config file
net = OrderedDict()
for l in layernames:
    pars = parser.items(l)
    name, layer = build_layer(pars)
    if layer:
        net[name] = layer

# Load data from csv
df = pd.read_csv('train_small.csv',index_col=0)
data = np.array(df.iloc[:,1:]).reshape(-1,28,28)
label = np.array(df.iloc[:,0])
N = data.shape[0]
nclass = np.unique(label).size
calc_wtop(net, N, label)
net['input'].top = data

# run gradient decent
for i in range(200):
    cost = forward_pass(net,0)
    print "%dth iteration: cost = %f" % (i, cost) 
    backward_pass(net)
    update_pars(net,0.001)




