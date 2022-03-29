#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from tensorflow.keras.regularizers import L1L2
from pyts.image import GramianAngularField
#%%
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
person_train_valid = np.load("person_train_valid.npy")
X_train_valid = np.load("X_train_valid.npy")
y_train_valid = np.load("y_train_valid.npy")
person_test = np.load("person_test.npy")

## Printing the shapes of the numpy arrays

print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
print ('Test data shape: {}'.format(X_test.shape))
print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
print ('Test target shape: {}'.format(y_test.shape))
print ('Person train/valid shape: {}'.format(person_train_valid.shape))
print ('Person test shape: {}'.format(person_test.shape))
y_train_valid -= 769
y_test -= 769
#%%
ind_valid = np.random.choice(X_train_valid.shape[0], 500, replace=False)
ind_train = np.array(list(set(range(X_train_valid.shape[0])).difference(set(ind_valid))))

# Creating the training and validation sets using the generated indices
(x_train, x_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
(y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
#%%
def data_prep(X,y,trim,sub_sample,average,noise,willMaxPool,willAverage,willSubsample):
    
    X = X[:,:,0:trim]
    print('Shape of X after trimming:',X.shape)
    
    total_X = np.zeros((1, X.shape[1], int(X.shape[2]/sub_sample)))
    total_y = np.zeros((1))
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)

    
    if willMaxPool:
        # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
        X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
        total_X = np.vstack((total_X,X_max))
        total_y = np.hstack((total_y, y))
        print('Shape of X after maxpooling:',total_X.shape)
    
    
    # Averaging + noise 
    if willAverage:
        X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
        if noise:
            X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
        total_X = np.vstack((total_X, X_average))
        total_y = np.hstack((total_y, y))
        print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    if willSubsample:
        for i in range(sub_sample):
            
            X_subsample = X[:, :, i::sub_sample] + \
                                (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
                
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
            
        
        print('Shape of X after subsampling and concatenating:',total_X.shape)
    
    total_X=total_X[1:,:,:]
    total_y=total_y[1:]
    return total_X,total_y

def fitTransformGramianTimeSeries(dataset,imgSize,timeStep):
    gcfs = {}
    amountOfTime = dataset.shape[2]
    noOfImages = int(amountOfTime / timeStep)
    amountOfFeatures = dataset.shape[1]
    amountOfSamples = dataset.shape[0]
    imageOut = np.empty((amountOfSamples,noOfImages,imgSize,imgSize, amountOfFeatures))
    for k in range(amountOfFeatures):
        feature = dataset[:,k,:]
        gcf = GramianAngularField(image_size=imgSize,method='d')
        gcf = gcf.fit(feature)
        string = "gcf"+str(k)
        gcfs[string] = gcf
        seperated = np.array_split(feature, noOfImages, axis=1)

        for ind in range(len(seperated)):
            section = seperated[ind]
            newImage = gcf.transform(section)
            imageOut[:,ind,:,:,k] = newImage

    return gcfs,imageOut

def transformGramianTimeStep(dataset,gcfs,imgSize,timeStep):
    amountOfTime = dataset.shape[2]
    noOfImages = int(amountOfTime / timeStep)
    amountOfFeatures = dataset.shape[1]
    amountOfSamples = dataset.shape[0]
    imageOut = np.empty((amountOfSamples,noOfImages,imgSize,imgSize, amountOfFeatures))
    for k in range(amountOfFeatures):
        feature = dataset[:,k,:]
        seperated = np.array_split(feature, noOfImages, axis=1)
        string = "gcf"+str(k)
        gcf = gcfs[string]
        for ind in range(len(seperated)):
            section = seperated[ind]
            newImage = gcf.transform(section)
            imageOut[:,ind,:,:,k] = newImage
    return imageOut
#%%
def testAcrossTrims(trimAmount,howMuchTimeWillImage,x_train,y_train,x_valid,y_valid,x_test,y_test):
    testResults = []
    for k in range(len(trimAmount)):
        
        trim = trimAmount[k]
        imageTime = howMuchTimeWillImage[k]
        
        X_train_prep,y_train_prep = data_prep(x_train,y_train,trim,2,2,
                                                           noise=True,willMaxPool=True,
                                                           willAverage=True,willSubsample=True)
        X_valid_prep,y_valid_prep = data_prep(x_valid,y_valid,trim,2,2,
                                                           noise=True,willMaxPool=True,
                                                           willAverage=True,willSubsample=True)

        y_trainCat = to_categorical(y_train_prep, 4)
        y_validCat = to_categorical(y_valid_prep, 4)

        gcfs,trainImages = fitTransformGramianTimeSeries(X_train_prep,30,imageTime)
        valImages = transformGramianTimeStep(X_valid_prep,gcfs,30,imageTime)
    
        l2Strength=0.1
        l1Strength=0.00
        kernelSize = 2
    
        inp = layers.Input((trainImages.shape[1], trainImages.shape[2], trainImages.shape[3], trainImages.shape[4]))
        hidden = TimeDistributed(layers.Conv2D(filters=8, 
                                           kernel_regularizer=L1L2(l1=l1Strength, l2=l2Strength),
                                           kernel_size=(kernelSize,kernelSize), padding='same', activation='elu'))(inp)
        hidden = TimeDistributed(layers.MaxPooling2D(pool_size=(2,2), padding='same'))(hidden)
        hidden = TimeDistributed(layers.BatchNormalization())(hidden)
        hidden = TimeDistributed(layers.Dropout(0.5))(hidden)
        hidden = TimeDistributed(layers.Conv2D(filters=10, 
                                           kernel_regularizer=L1L2(l1=l1Strength, l2=l2Strength),
                                           kernel_size=(kernelSize,kernelSize), padding='same', activation='elu'))(inp)
        hidden = TimeDistributed(layers.MaxPooling2D(pool_size=(2,2), padding='same'))(hidden)
        hidden = TimeDistributed(layers.BatchNormalization())(hidden)
        hidden = TimeDistributed(layers.Dropout(0.5))(hidden)
        hidden = TimeDistributed(layers.Flatten())(hidden)
        #hidden = layers.Bidirectional(layers.LSTM(64,return_sequences=True))(hidden)
        hidden = layers.Bidirectional(layers.LSTM(32,return_sequences=False))(hidden)
        hidden = layers.Dropout(0.5)(hidden)
        hidden = layers.Dense(4, kernel_regularizer=L1L2(l1=l1Strength, l2=l2Strength),activation='softmax')(hidden)
        
        model = tf.keras.models.Model(inputs=inp, outputs=hidden)
        model.summary()
    
        epochs = 20
        optimizer = keras.optimizers.Adam(lr=0.001,epsilon=1e-8, decay=0.001)
        
        model.compile(loss='categorical_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    
        # Training and validating the model
        results = model.fit(trainImages,
                 y_trainCat,
                 batch_size=32,
                 epochs=epochs,
                 validation_data=(valImages, y_validCat), verbose=True)

        X_test_prep,y_test_prep = data_prep(X_test,y_test,trim,2,2,noise=False,willMaxPool=False,
                                                               willAverage=True,willSubsample=False)
        y_testCat = to_categorical(y_test_prep, 4)
    
        testImages = transformGramianTimeStep(X_test_prep,gcfs,30,imageTime)
    
        modelScore = model.evaluate(testImages, y_testCat, verbose=1)
        testResults.append(modelScore[1])
        print('Test accuracy of the hybrid CNN-LSTM model:',modelScore[1])

#%%
trimAmount = [500,750]
howMuchTimeWillImage = [50,75]

testResults = testAcrossTrims(trimAmount,howMuchTimeWillImage,x_train,y_train,x_valid,y_valid,X_test,y_test)


#41.0 for 500, 43.1 for 750























