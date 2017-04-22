
# coding: utf-8

# # Bayesian Optimization on Keras

# ### MNIST training on Keras with Bayesian optimization
# * This notebook runs MNIST training on Keras using Bayesian optimization to find the best hyper parameters.
# * The MNIST model here is just a simple one with one input layer, one hidden layer and one output layer, without convolution.
# * Hyperparameters of the model include the followings:
# * - output shape of the first layer
# * - dropout rate of the first layer
# * - output shape of the second layer
# * - dropout rate of the second layer
# * - batch size
# * - number of epochs
# * - validation rate
# * I used GPy and GPyOpt to run Bayesian optimization.

# #### Import libraries

# In[21]:

import GPy, GPyOpt
import numpy as np
import pandas as pds
import random
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# In[ ]:




# #### Define MNIST model
# * includes data loading function, training function, fit function and evaluation function 

# In[22]:

# MNIST class
class MNIST():
    def __init__(self, first_input=784, last_output=10,
                 l1_out=512, 
                 l2_out=512, 
                 l1_drop=0.2, 
                 l2_drop=0.2, 
                 batch_size=100, 
                 epochs=10, 
                 validation_split=0.1):
        self.__first_input = first_input
        self.__last_output = last_output
        self.l1_out = l1_out
        self.l2_out = l2_out
        self.l1_drop = l1_drop
        self.l2_drop = l2_drop
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.mnist_data()
        self.__model = self.mnist_model()
        
    # load mnist data from keras dataset
    def mnist_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)
        return X_train, X_test, Y_train, Y_test
    
    # mnist model
    def mnist_model(self):
        model = Sequential()
        model.add(Dense(self.l1_out, input_shape=(self.__first_input,)))
        model.add(Activation('relu'))
        model.add(Dropout(self.l1_drop))
        model.add(Dense(self.l2_out))
        model.add(Activation('relu'))
        model.add(Dropout(self.l2_drop))
        model.add(Dense(self.__last_output))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        return model
    
    # fit mnist model
    def mnist_fit(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)
        
        self.__model.fit(self.__x_train, self.__y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=0,
                       validation_split=self.validation_split,
                       callbacks=[early_stopping])
    
    # evaluate mnist model
    def mnist_evaluate(self):
        self.mnist_fit()
        
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=0)
        return evaluation


# In[ ]:




# #### Runner function for the MNIST model

# In[11]:

# function to run mnist class
def run_mnist(first_input=784, last_output=10,
              l1_out=512, l2_out=512, 
              l1_drop=0.2, l2_drop=0.2, 
              batch_size=100, epochs=10, validation_split=0.1):
    
    _mnist = MNIST()
    mnist_evaluation = _mnist.mnist_evaluate()
    return mnist_evaluation


# In[ ]:




# ## Bayesian Optimization
# #### bounds for hyper parameters

# In[12]:

# bounds for hyper-parameters in mnist model
# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'validation_split', 'type': 'continuous',  'domain': (0.0, 0.3)},
          {'name': 'l1_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
          {'name': 'l2_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
          {'name': 'l1_out',           'type': 'discrete',    'domain': (64, 128, 256, 512, 1024)},
          {'name': 'l2_out',           'type': 'discrete',    'domain': (64, 128, 256, 512, 1024)},
          {'name': 'batch_size',       'type': 'discrete',    'domain': (10, 100, 500)},
          {'name': 'epochs',           'type': 'discrete',    'domain': (5, 10, 20)}]


# #### Bayesian Optimization

# In[13]:

# function to optimize mnist model
def f(x):
    print(x)
    evaluation = run_mnist(
        l1_out = int(x[:,1]), 
        l2_out = int(x[:,2]), 
        l1_drop = float(x[:,3]),
        l2_drop = float(x[:,4]), 
        batch_size = int(x[:,5]), 
        epochs = int(x[:,6]), 
        validation_split = float(x[:,0]))
    print("loss:{0} \t\t accuracy:{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]


# #### Optimizer instance

# In[14]:

# optimizer
opt_mnist = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)


# #### Running optimization

# In[15]:

# optimize mnist model
opt_mnist.run_optimization(max_iter=10)


# #### The output

# In[16]:

# print optimized mnist model
print("optimized parameters: {0}".format(opt_mnist.x_opt))
print("optimized loss: {0}".format(opt_mnist.fx_opt))


# In[ ]:




# In[ ]:



