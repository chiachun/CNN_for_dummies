

# layer names in the config file
# layernames = ['input','conv1','relu1','maxpool1','softmax']
# parser = SafeConfigParser()
# parser.optionxform = str  # make option names case sensitive
# parser.read('config.ini')

# # Read parameters from the config file
# net = OrderedDict()
# for l in layernames:
#     pars = parser.items(l)
#     name, layer = build_layer(pars)
#     if layer:
#         net[name] = layer


def predict(net,N):
    for i,l in enumerate(net.itervalues()):
        l.N = N
        if l.ltype!='softmax': l.forward()
        if l.ltype=='softmax':
            prob = l.predict()
            return prob


dft = pd.read_csv('test_small.csv',index_col=0)
yt = np.array(dft.iloc[:,0])
data = np.array(dft.iloc[:1000,1:]).reshape(-1,28,28)
N = data.shape[0]
net['input'].top = data
res = predict(net,N)
accuracy = np.sum(yt - np.argmax(res, axis=1)) / 1000. + 1
