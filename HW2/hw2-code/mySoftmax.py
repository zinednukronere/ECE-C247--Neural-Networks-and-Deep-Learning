#%%
import random
import numpy as np
from utils.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from nndl import Softmax

#%%
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'dataset\cifar-10-batches-py' # You need to update this line
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


#%%
# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)


#%%
np.random.seed(1)

num_classes = len(np.unique(y_train))
num_features = X_train.shape[1]

softmax = Softmax(dims=[num_classes, num_features])
#%%
weights = softmax.W
multiplication = weights.dot(X_train.T).T
exponentials = np.exp(multiplication)
exponentialSum = np.sum(exponentials,axis=1)
losses = []
for i in range(0,len(exponentialSum)):
    exponentials[i]=exponentials[i]/exponentialSum[i]
    label = y_train[i]
    crossEntropy = -np.log(exponentials[i][label])
    losses.append(crossEntropy)
out = sum(losses)/len(losses)
    
#%%
multiplication = weights.dot(X_train.T).T
exp = np.exp(multiplication)
for i in range(len(X_train)):
        exp[i] /= np.sum(exp[i])

yHot = np.zeros([y_train.shape[0],np.max(y_train)+1])
for k in range(yHot.shape[0]):
    yHot[k][y_train[k]]=1

gradient = np.dot(X_train.T, (exp - yHot))

#%%
loss = 0.0
grad = np.zeros_like(weights)
  
# ================================================================ #
# YOUR CODE HERE:
#   Calculate the softmax loss and the gradient. Store the gradient
#   as the variable grad.
# ================================================================ #
losses=[]
for index in range(0,X_train.shape[0]):  
    data = X_train[index]
    multiplication = weights.dot(data.T).T
    exponentials = np.exp(multiplication)
    exponentialSum = np.sum(exponentials)
    exponentials=exponentials/exponentialSum
    label = y_train[index]
    crossEntropy = -np.log(exponentials[label])
    losses.append(crossEntropy)
    yHot = np.zeros([1,np.max(y_train)+1])
    yHot[0,label]=1
    data = np.reshape(data,[1,data.shape[0]])
    exponentials = np.reshape(exponentials,[1,exponentials.shape[0]])
    derivative = np.dot(data.T, (exponentials - yHot))
    grad=grad+derivative.T
loss = sum(losses)/len(losses)
grad = grad/[X_train.shape[0]]
    
#%%
multiplication =weights.dot(X_train.T).T
soft = np.exp(multiplication)
sums = np.sum(soft,axis=1)
probs = (soft.T / sums).T
predsForClass = probs[np.arange(y_train.size),y_train]
loss = -np.log(predsForClass)
loss = np.mean(loss)
yHot = np.zeros([y_train.shape[0],9+1])
yHot[np.arange(y_train.size),y_train] = 1
grad = (1/X_train.shape[0])*np.dot(X_train.T, (probs - yHot)).T


#%%
    multiplication =self.W.dot(X.T).T
    soft = np.exp(multiplication)
    for i in range(X.shape[0]):
        soft[i] /= np.sum(soft[i])
    yHot = np.zeros([y.shape[0],np.max(y)+1])
    for k in range(yHot.shape[0]):
        yHot[k][y[k]]=1
    

    grad = (1/X.shape[0])*np.dot(X.T, (soft - yHot)).T
    loss = self.loss(X,y)
#%%
softmax = Softmax(dims=[num_classes, num_features])
rates = [10**i for i in range(-3,0)]

valAccuracies = []
valLosses = []
for rate in rates:
    softmax.train(X_train, y_train, learning_rate=rate, num_iters=1500, verbose=False)
    valLoss,a = softmax.fast_loss_and_grad(X_val, y_val)
    valPreds = softmax.predict(X_val)
    valAcc = np.mean(np.equal(y_val, valPreds)) 
    valAccuracies.append(valAcc)
    valLosses.append(valLoss)
    print('Current rate:',rate,'validation accuracy: {}'.format(valAcc),'validation Loss: {}'.format(valLoss))
    print("Best validation loss so far {}".format(min(valLosses)))
print("-"*20)
#%%
bestIndex = np.argmin(valLosses)
bestLR = rates[bestIndex]
softmax.train(X_train, y_train, learning_rate=bestLR, num_iters=1500, verbose=False)
yTestPred = softmax.predict(X_test)
testAcc = np.mean(np.equal(y_test, yTestPred))
print('Best learning rate is',bestLR, 'Test Set Error Rate is', 1-testAcc) 















