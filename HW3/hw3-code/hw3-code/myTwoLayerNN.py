import numpy as np
#%%
input_size = 4
hidden_size = 10
output_size = 3
num_inputs = 5
std=1e-4
params = dict()
params['W1'] = std * np.random.randn(hidden_size, input_size)
params['b1'] = np.zeros(hidden_size)
params['W2'] = std * np.random.randn(output_size, hidden_size)
params['b2'] = np.zeros(output_size)
#%%
def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y
X, y = init_toy_data()

#%%


#%%
W1, b1 = net.params['W1'], net.params['b1']
W2, b2 = net.params['W2'], net.params['b2']
b1 = b1.reshape([len(b1),1])
b2 = b2.reshape([len(b2),1])
layer1Out = W1.dot(X.T) +  b1
ReluOut = np.maximum(0,layer1Out)
layer2Out = W2.dot(ReluOut) +  b2
scores = layer2Out.T
#%%
#from layers import *
#from layer_utils import *
loss, dz = softmax_loss(scores, y)
loss += 0.5*0*(np.sum(W1*W1) + np.sum(W2*W2))
#%%
soft = np.exp(scores)
sums = np.sum(soft,axis=1)
probs = soft / sums[:,None]
predsForClass = probs[np.arange(y.size),y]
SoftmaxLoss = np.mean(-np.log(predsForClass))
l2regularization = np.sum(W1*W1) + np.sum(W2*W2)
l2regularization = 0.5*0.05*l2regularization
loss=SoftmaxLoss+l2regularization
print(loss)
#%%
np.random.seed(0)
net = TwoLayerNet(input_size, hidden_size, output_size, std=1e-1)
#%%
loss, _ = net.loss(X, y, reg=0.05)
print(loss)
#%%
grad = probs.copy()
grad[np.arange(y.size),y] -= 1
dLA3=grad/X.shape[0]
dA3W2=ReluOut.copy()
b2grad = np.sum(dLA3,axis=0)
w2grad = (dA3W2.dot(dLA3)).T
dA3A2 = W2
dA2dA1 = layer1Out.copy()
dA2dA1[dA2dA1<0]=0
dA2dA1[dA2dA1>0]=1
dA1dW1 = X
kronecker = ((dLA3.dot(dA3A2)).T*dA2dA1)
b1grad = np.sum(kronecker.T,axis=0)
w1grad = (kronecker.dot(dA1dW1))

#%%
loss, grads = net.loss(X, y, reg=0.05)

