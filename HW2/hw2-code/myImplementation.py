def compute_L2_distances_vectorized(X_train,X):
  """
  Compute the distance between each test point in X and each training point
  in self.X_train WITHOUT using any for loops.

  Inputs:
  - X: A numpy array of shape (num_test, D) containing test data.

  Returns:
  - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
    is the Euclidean distance between the ith test point and the jth training
    point.
  """
  num_test = X.shape[0]
  num_train = X_train.shape[0]
  dists = np.zeros((num_test, num_train))

  # ================================================================ #
  # YOUR CODE HERE:
  #   Compute the L2 distance between the ith test point and the jth       
  #   training point and store the result in dists[i, j].  You may 
  #   NOT use a for loop (or list comprehension).  You may only use
  #   numpy operations.
  #
  #   HINT: use broadcasting.  If you have a shape (N,1) array and
  #   a shape (M,) array, adding them together produces a shape (N, M) 
  #   array.
  # ================================================================ #

  trainsquared = np.sum(X_train**2, axis=1, keepdims=True)
  testsquared = np.sum(X**2, axis=1)
  multiplied = np.dot(X_train, X.T)
  dists = np.sqrt(trainsquared - 2*multiplied + testsquared)
  dists = dists.T
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dists
#%%
import numpy as np # for doing most of our calculations
import matplotlib.pyplot as plt# for plotting
from utils.data_utils import load_CIFAR10 # function to load the CIFAR-10 dataset.
#%%
cifar10_dir = 'dataset\cifar-10-batches-py' # You need to update this line
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
#%%
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)
#%%
knn = KNN()
knn.train(X=X_train, y=y_train)
distances = knn.compute_L2_distances_vectorized(X=X_train)
#%%
predictions = knn.predict_labels(distances,1)
#%%
num_folds = 5

X_train_folds = []
y_train_folds =  []
np.random.seed(42)
# ================================================================ #
# YOUR CODE HERE:
#   Split the training data into num_folds (i.e., 5) folds.
#   X_train_folds is a list, where X_train_folds[i] contains the 
#      data points in fold i.
#   y_train_folds is also a list, where y_train_folds[i] contains
#      the corresponding labels for the data in X_train_folds[i]
# ================================================================ #
indices = np.arange(0,X_train.shape[0])
randomIndices=np.random.permutation(indices)
foldSize = int(X_train.shape[0]/num_folds)
start = 0
end = start + foldSize
for i in range(0,num_folds-1):
    indices = randomIndices[start:end]
    foldtraindata=X_train[indices]
    foldtrainlabel = y_train[indices]
    X_train_folds.append(foldtraindata)
    y_train_folds.append(foldtrainlabel)
    start = end
    end = start+foldSize
indices = randomIndices[start:]
foldtraindata=X_train[indices]
foldtrainlabel = y_train[indices]
X_train_folds.append(foldtraindata)
y_train_folds.append(foldtrainlabel)
#%%
trainCopy = X_train_folds.copy()
trainLabelsCopy = y_train_folds.copy()
foldTest = trainCopy[0]
foldTtestlabel = trainLabelsCopy[0]
trainCopy.pop(0)
trainLabelsCopy.pop(0)
foldTr=np.array(trainCopy)
foldTrLab=np.array(trainLabelsCopy)
foldTr = foldTr.reshape([foldTr.shape[0]*foldTr.shape[1],foldTr.shape[2]])
#%%
for i in range(0,len(X_train_folds)):
    trainCopy = X_train_folds.copy()
    trainLabelsCopy = y_train_folds.copy()
    foldTest = trainCopy[i]
    foldTtestlabel = trainLabelsCopy[i]
    trainCopy.pop(i)
    trainLabelsCopy.pop(i)
    foldTr=np.array(trainCopy)
    foldTrLab=np.array(trainLabelsCopy)
    foldTr = foldTr.reshape([foldTr.shape[0]*foldTr.shape[1],foldTr.shape[2]])
    foldTrLab = foldTrLab.reshape([foldTrLab.shape[0]*foldTrLab.shape[1]])
    knn.train(foldTr,foldTrLab)
    distances = knn.compute_L2_distances_vectorized(X=foldTest)
    predictions = knn.predict_labels(distances,1)
    error = sum(predictions!=foldTtestlabel)/len(foldTtestlabel)        
    #foldError.append(error)

#%%
ks = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]


kError = []
for k in ks:
    foldError = []
    for i in range(0,len(X_train_folds)):
        trainCopy = X_train_folds.copy()
        trainLabelsCopy = y_train_folds.copy()
        foldTest = trainCopy[i]
        foldTtestlabel = trainLabelsCopy[i]
        trainCopy.pop(i)
        trainLabelsCopy.pop(i)
        foldTr=np.array(trainCopy)
        foldTrLab=np.array(trainLabelsCopy)
        foldTr = foldTr.reshape([foldTr.shape[0]*foldTr.shape[1],foldTr.shape[2]])
        foldTrLab = foldTrLab.reshape([foldTrLab.shape[0]*foldTrLab.shape[1]])
        knn = KNN()
        knn.train(foldTr,foldTrLab)
        distances = knn.compute_L2_distances_vectorized(X=foldTest)
        predictions = knn.predict_labels(distances,k=1000)
        error = sum(predictions!=foldTtestlabel)/len(foldTtestlabel)
        print("Done fold %d for k %d. Error is %.2f"%(i,k,error))
        foldError.append(error)
    kError.append(np.mean(foldError))    

#%%
#time_start =time.time()

L1_norm = lambda x: np.linalg.norm(x, ord=1)
L2_norm = lambda x: np.linalg.norm(x, ord=2)
Linf_norm = lambda x: np.linalg.norm(x, ord= np.inf)
norms = [L1_norm, L2_norm, Linf_norm]
normNames=["l1","l2","inf"]
# ================================================================ #
# YOUR CODE HERE:
#   Calculate the cross-validation error for each norm in norms, testing
#   the trained model on each of the 5 folds.  Average these errors
#   together and make a plot of the norm used vs the cross-validation error
#   Use the best cross-validation k from the previous part.  
#
#   Feel free to use the compute_distances function.  We're testing just
#   three norms, but be advised that this could still take some time.
#   You're welcome to write a vectorized form of the L1- and Linf- norms
#   to speed this up, but it is not necessary.
# ================================================================ #
bestK=30
knn = KNN()
kError = []
for ind in range(0,len(norms)):
    normType = norms[ind]
    normName = normNames[ind]
    foldError = []
    for i in range(0,len(X_train_folds)):
        trainCopy = X_train_folds.copy()
        trainLabelsCopy = y_train_folds.copy()
        foldTest = trainCopy[i]
        foldTtestlabel = trainLabelsCopy[i]
        trainCopy.pop(i)
        trainLabelsCopy.pop(i)
        foldTr=np.array(trainCopy)
        foldTrLab=np.array(trainLabelsCopy)
        foldTr = foldTr.reshape([foldTr.shape[0]*foldTr.shape[1],foldTr.shape[2]])
        foldTrLab = foldTrLab.reshape([foldTrLab.shape[0]*foldTrLab.shape[1]])
        knn.train(foldTr,foldTrLab)
        if normName=="l1":
           distances = knn.compute_L1_distances_vectorized(foldTest)
        elif normName=="l2":
           distances = knn.compute_L2_distances_vectorized(foldTest)
        else:
           distances = knn.compute_Linf_distances_vectorized(foldTest)
                
        #distances = knn.compute_distances(X=foldTest,norm=normType)
        predictions = knn.predict_labels(distances,k)
        error = sum(predictions!=foldTtestlabel)/len(foldTtestlabel)
        print("ye")
        foldError.append(error)
    print("Done cross validation for norm %s. Error is %.2f"%(normName,np.mean(foldError)))
    kError.append(np.mean(foldError))    
# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #
print('Computation time: %.2f'%(time.time()-time_start))

