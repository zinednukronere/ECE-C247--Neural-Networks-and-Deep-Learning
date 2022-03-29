#%%
import numpy as np
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
person_train_valid = np.load("person_train_valid.npy")
X_train_valid = np.load("X_train_valid.npy")
y_train_valid = np.load("y_train_valid.npy")
person_test = np.load("person_test.npy")

print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
print ('Test data shape: {}'.format(X_test.shape))
print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
print ('Test target shape: {}'.format(y_test.shape))
print ('Person train/valid shape: {}'.format(person_train_valid.shape))
print ('Person test shape: {}'.format(person_test.shape))
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
plt.title('EEG-as-an-image\nFirst 10 Trials\n22*10 Channels Stacked Vertically')
plt.imshow(np.vstack(X_train_valid[0:10]))
#plt.imshow(np.vstack(X_train_valid[0:10]),cmap='inferno')
plt.colorbar()
plt.show()

plt.figure(figsize=(17,4))
plt.title('Example of a EEG signal over 1000 time bins')
plt.plot(X_train_valid[0,0],label='EEG - Channel 0')
plt.plot(X_train_valid[0,1],label='EEG - Channel 1')
plt.plot(X_train_valid[0,21],label='EEG - Channel 21')
plt.legend()
plt.xlabel('label: ' + str(y_train_valid[0]))
plt.grid()
plt.show()

#%%
## Loading and visualizing the data

## Loading the dataset


X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
person_train_valid = np.load("person_train_valid.npy")
X_train_valid = np.load("X_train_valid.npy")
y_train_valid = np.load("y_train_valid.npy")
person_test = np.load("person_test.npy")

## Adjusting the labels so that 

# Cue onset left - 0
# Cue onset right - 1
# Cue onset foot - 2
# Cue onset tongue - 3

y_train_valid -= 769
y_test -= 769

## Visualizing the data

ch_data = X_train_valid[:,8,:] # Extracting channel 9 data

## Extract the mean eeg data for label 0, 1, 2, 3
class_0_ind = np.where(y_train_valid == 0) # Extracting the indices corresponding to label 0
ch_data_class_0 = ch_data[class_0_ind] # Extracting the eeg data for label 0
avg_ch_data_class_0 = np.mean(ch_data_class_0,axis=0) # Extracting the mean eeg data for label 0


class_1_ind = np.where(y_train_valid == 1)
ch_data_class_1 = ch_data[class_1_ind]
avg_ch_data_class_1 = np.mean(ch_data_class_1,axis=0)

class_2_ind = np.where(y_train_valid == 2)
ch_data_class_2 = ch_data[class_2_ind]
avg_ch_data_class_2 = np.mean(ch_data_class_2,axis=0)

class_3_ind = np.where(y_train_valid == 3)
ch_data_class_3 = ch_data[class_3_ind]
avg_ch_data_class_3 = np.mean(ch_data_class_3,axis=0)


plt.plot(np.arange(1000),avg_ch_data_class_0)
plt.plot(np.arange(1000),avg_ch_data_class_1)
plt.plot(np.arange(1000),avg_ch_data_class_2)
plt.plot(np.arange(1000),avg_ch_data_class_3)
plt.axvline(x=500, label='line at t=500',c='cyan')

plt.legend(["Cue Onset left", "Cue Onset right", "Cue onset foot", "Cue onset tongue"])

#%%
def data_prep(X,y,sub_sample,average,noise):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:500]
    print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    print('Shape of X after subsampling and concatenating:',total_X.shape)
    return total_X,total_y


X_train_valid_prep,y_train_valid_prep = data_prep(X_train_valid,y_train_valid,2,2,True)


#%%
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras
import numpy as np

#%%
## Preprocessing the dataset

X_train_valid_prep,y_train_valid_prep = data_prep(X_train_valid,y_train_valid,2,2,True)
X_test_prep,y_test_prep = data_prep(X_test,y_test,2,2,True)

print('lel')
print(X_train_valid_prep.shape)
print(y_train_valid_prep.shape)
print(X_test_prep.shape)
print(y_test_prep.shape)
print('lel')



## Random splitting and reshaping the data

# First generating the training and validation indices using random splitting
ind_valid = np.random.choice(8460, 1500, replace=False)
ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))

# Creating the training and validation sets using the generated indices
(x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid] 
(y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
print('Shape of training set:',x_train.shape)
print('Shape of validation set:',x_valid.shape)
print('Shape of training labels:',y_train.shape)
print('Shape of validation labels:',y_valid.shape)


# Converting the labels to categorical variables for multiclass classification
print('lul')
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)
print('lul')
y_train = to_categorical(y_train, 4)
y_valid = to_categorical(y_valid, 4)
y_test = to_categorical(y_test_prep, 4)
print('Shape of training labels after categorical conversion:',y_train.shape)
print('Shape of validation labels after categorical conversion:',y_valid.shape)
print('Shape of test labels after categorical conversion:',y_test.shape)

# Adding width of the segment to be 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)
print('Shape of training set after adding width info:',x_train.shape)
print('Shape of validation set after adding width info:',x_valid.shape)
print('Shape of test set after adding width info:',x_test.shape)

#%%
# CNN model with residual connections

#X_train_valid = tf.keras.utils.normalize(X_train_valid)
#X_test = tf.keras.utils.normalize(X_test)

#X_train, X_val, y_train, y_val = train_test_split(X_train_valid, y_train_valid, test_size=0.2, random_state=42)
#X_train = X_train[:,:,:,np.newaxis]
#X_val = X_val[:,:,:,np.newaxis]
#X_test = X_test[:,:,:,np.newaxis]
#y_train = to_categorical(y_train - 769)
#y_val = to_categorical(y_val - 769)
#y_test = to_categorical(y_test - 769)

(num_eeg_channels, num_time_points) = (22, 250)


inputs = keras.Input(shape=(num_eeg_channels, num_time_points, 1))
x = layers.Conv2D(25, (1, 11), input_shape=(num_eeg_channels, num_time_points, 1),data_format='channels_last')(inputs)
x = layers.Conv2D(50, (22, 1))(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x_res1 = layers.BatchNormalization()(x)
x = layers.Conv2D(50, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(x_res1)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(50, (1, 10), activation=None,kernel_regularizer=regularizers.l2(0.001),padding='same')(x)
x = layers.Add()([x, x_res1])
x = layers.Activation('elu')(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x = layers.Conv2D(100, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)
x_res2 = layers.BatchNormalization()(x)
x = layers.Conv2D(100, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(x_res2)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(100, (1, 10), activation=None,kernel_regularizer=regularizers.l2(0.001),padding='same')(x)
x = layers.Add()([x, x_res2])
x = layers.Activation('elu')(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x = layers.Conv2D(200, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
outputs = layers.Dense(4,kernel_regularizer=regularizers.l2(0.001),activation='softmax')(x)
res_cnn_model = keras.Model(inputs, outputs)
res_cnn_model.summary()

res_cnn_model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['categorical_accuracy'])
history = res_cnn_model.fit(x_train, y_train, epochs=2000, 
                validation_data=(x_valid, y_valid))


#%%
# CNN +LSTM with residual connections
(num_eeg_channels, num_time_points) = (22, 250)

inputs = keras.Input(shape=(num_eeg_channels, num_time_points, 1))
x = layers.Conv2D(25, (1, 11), input_shape=(num_eeg_channels, num_time_points, 1),data_format='channels_last')(inputs)
x = layers.Conv2D(50, (22, 1))(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x_res1 = layers.BatchNormalization()(x)
x = layers.Conv2D(50, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(x_res1)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(50, (1, 10), activation=None,kernel_regularizer=regularizers.l2(0.001),padding='same')(x)
x = layers.Add()([x, x_res1])
x = layers.Activation('elu')(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x = layers.Conv2D(100, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)
x_res2 = layers.BatchNormalization()(x)
x = layers.Conv2D(100, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(x_res2)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(100, (1, 10), activation=None,kernel_regularizer=regularizers.l2(0.001),padding='same')(x)
x = layers.Add()([x, x_res2])
x = layers.Activation('elu')(x)
x = layers.MaxPool2D((1, 3))(x)
x = layers.Dropout(0.55)(x)
x = layers.Conv2D(200, (1, 10), activation='elu',kernel_regularizer=regularizers.l2(0.001),padding='same')(x)
x = layers.MaxPool2D((1, 2))(x)
x = layers.Dropout(0.55)(x)
x = layers.BatchNormalization()(x)


x = layers.Permute((2, 3, 1))(x)
x = layers.TimeDistributed(layers.Flatten())(x)
x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(32))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(4,kernel_regularizer=regularizers.l2(0.001),activation='softmax')(x)

res_lstm_model = keras.Model(inputs, outputs)
res_lstm_model.summary()

res_lstm_model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['categorical_accuracy'])

history = res_lstm_model.fit(x_train, y_train, epochs=2000, 
                validation_data=(x_test, y_test))


