#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from keras.layers import TimeDistributed
from tensorflow.keras.regularizers import L1L2

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
ind_valid = np.random.choice(2115, 500, replace=False)
ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))

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


#%%
from tensorflow.python.keras import regularizers

def testAcrossTrims(trimAmount,x_train,y_train,x_valid,y_valid,x_test,y_test):
    testResults = []
    for k in range(len(trimAmount)):
        
        trim = trimAmount[k]
        
        
        X_train_prep,y_train_prep = data_prep(x_train,y_train,trim,2,2,
                                                           noise=True,willMaxPool=True,
                                                           willAverage=True,willSubsample=True)
        X_valid_prep,y_valid_prep = data_prep(x_valid,y_valid,trim,2,2,
                                                           noise=True,willMaxPool=True,
                                                           willAverage=True,willSubsample=True)
                                                           
        
        '''X_train_prep,y_train_prep = x_train[:,:,0:trim],y_train
        X_valid_prep,y_valid_prep = x_valid[:,:,0:trim],y_valid'''

        y_trainCat = to_categorical(y_train_prep, 4)
        y_validCat = to_categorical(y_valid_prep, 4)

        x_trainExt = np.expand_dims(X_train_prep, axis=3)
        x_validExt = np.expand_dims(X_valid_prep, axis=3)
        
        
        inputs = keras.Input(shape=(x_trainExt.shape[1], x_trainExt.shape[2], 1))
        conv1 = layers.Conv2D(25, (1, 11), input_shape=(x_trainExt.shape[1], x_trainExt.shape[2], 1),data_format='channels_last')(inputs)

        conv2 = layers.Conv2D(50, (22, 1))(conv1)
        max1 = layers.MaxPool2D((1, 3))(conv2)
        drop1 = layers.Dropout(0.55)(max1)

        conv3 = layers.Conv2D(50, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(drop1)
        drop2 = layers.Dropout(0.55)(conv3)
        bn1 = layers.BatchNormalization()(drop2)

        conv4 = layers.Conv2D(50, (1, 10), activation=None,kernel_regularizer=regularizers.l2(0.001),padding='same')(bn1)
        conv4 = layers.Activation('elu')(conv4)
        max2 = layers.MaxPool2D((1, 3))(conv4)
        drop3 = layers.Dropout(0.55)(max2)

        conv5 = layers.Conv2D(100, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(drop3)
        max3 = layers.MaxPool2D((1, 3))(conv5)
        drop4 = layers.Dropout(0.55)(max3)
        bn2 = layers.BatchNormalization()(drop4)

        conv6 = layers.Conv2D(100, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(bn2)
        drop5 = layers.Dropout(0.55)(conv6)
        bn3 = layers.BatchNormalization()(drop5)

        conv7 = layers.Conv2D(100, (1, 10), activation=None,kernel_regularizer=regularizers.l2(0.001),padding='same')(bn3)
        conv7 = layers.Activation('elu')(conv7)
        max4 = layers.MaxPool2D((1, 3))(conv7)
        drop6 = layers.Dropout(0.55)(max4)

        conv8 = layers.Conv2D(200, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(drop6)
        max5 = layers.MaxPool2D((1, 2))(conv8)
        drop7 = layers.Dropout(0.55)(max5)
        bn4 = layers.BatchNormalization()(drop7)

        flat = layers.Flatten()(bn4)
        outputs = layers.Dense(4,kernel_regularizer=regularizers.l2(0.001),activation='softmax')(flat)
        cnn_model = keras.Model(inputs, outputs)
        cnn_model.summary()

        cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

        results = cnn_model.fit(x_trainExt, y_trainCat, epochs=250, 
                        validation_data=(x_validExt, y_validCat))
    
        
        # plot
        import matplotlib.pyplot as plt
        plt.plot(results.history['categorical_accuracy'])
        plt.plot(results.history['val_categorical_accuracy'])
        plt.title('Basic FC model accuracy trajectory')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        
        X_test_prep,y_test_prep = data_prep(X_test,y_test,trim,2,2,noise=False,willMaxPool=False,
                                                               willAverage=True,willSubsample=False)
        
        '''X_test_prep,y_test_prep = X_test[:,:,0:trim],y_test'''
        
        y_testCat = to_categorical(y_test_prep, 4)
    
        x_testExp = np.expand_dims(X_test_prep, axis=3)
    
        modelScore = cnn_model.evaluate(x_testExp, y_testCat, verbose=1)
        testResults.append(modelScore[1])
        print('Test accuracy of the hybrid CNN-LSTM model:',modelScore[1])

        return testResults
    
#%%
trimAmount = [750]

testResults = testAcrossTrims(trimAmount,x_train,y_train,x_valid,y_valid,X_test,y_test)
#%%
print(testResults)

























