This repository contains the code prepared for the course ECE C247: Neural Networks and Deep Learning taken during the Winter 2022 quarter at UCLA

HW1: Introduction to Jupyter Notebook
HW2: KNN and Softmax Classifier Implementation
HW3: Deriving backpropagation updates for autoencoders, implementing a fully connected network
HW4: Implementing different optimizers, batch normalization, dropout
HW5: Implement CNN, Spatial BatchNorm and model optimization

Project: 
In the project, our goal is to train various neural network based classifiers for classifying EEG data from BCI
Competition dataset IV. To this end, we consider 4 different models: CNN’s, CNN’s with residual connections, CNN’s
combined with LSTM, and CNN’s combined with LSTM’s with residual connections. As an added approach, we utilize Gramian Angular Field method to preprocess the data and using the resulting images as input for the CNN combined with LSTM model. We compare the performance of
our approaches with and without data augmentation across different objectives and different data lengths. In the end, the architecture that utilizes residual CNNs obtains the best results.
