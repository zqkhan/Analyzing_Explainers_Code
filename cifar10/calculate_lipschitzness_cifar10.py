#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


import numpy as np

import numpy as np
import pickle
from datetime import datetime
import time
from scipy.spatial.distance import pdist
import mat73

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.explanations import calculate_prob_lipschitz
import matplotlib.pyplot as plt

from tensorflow.python.keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical


# In[3]:


BATCH_SIZE = 32
epochs = 2
calculate = True
np.random.seed(0)


# In[4]:


for datatype in ['cifar1006']:
    save_lipschitz = 'plots/blackbox_' + datatype + '_lipschitz.pk'
    classifiers = ['2layer','4layer','linear','svm']
    L_range = np.arange(0, 1.1, 0.1)
    total_lipschitz = np.zeros(shape=(len(classifiers), len(L_range)))

    cifar = pickle.load(open('data/cifar10_6_6_train.pkl', 'rb'))
    data = cifar[1]
    labels = cifar[0]
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=42)
    x_train, x_val = np.array(x_train), np.array(x_val)
    x_train, x_val = x_train.reshape((len(x_train), -1)), x_val.reshape((len(x_val), -1))
    y_train_orig, y_val_orig = y_train.copy(), y_val.copy()
    y_train, y_val = to_categorical(y_train), to_categorical(y_val)
    input_shape = x_train.shape[-1]


    median_rad =np.median(pdist(x_train))

    activation = 'relu'

    model_input = Input(shape=(input_shape,), dtype='float32')

    net = Dense(32, activation=activation, name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)

    preds = Dense(10, activation='softmax', name='dense3',
                  kernel_regularizer=regularizers.l2(1e-3))(net)
    model = Model(model_input, preds)

    model.load_weights('models/' + datatype + '_blackbox.hdf5',
                           by_name=True)
    pred_model = Model(model_input, preds)
    pred_model.compile(loss=None,
                       optimizer='rmsprop',
                       metrics=None)

    if calculate:
        total_lipschitz[0, :] = calculate_prob_lipschitz(x_val, pred_model,
                                                   r=median_rad,
                                                   L_range=L_range,
                                                   num_points=len(x_val))

    del pred_model

    ###

    print("Training classifier with extra layer")

    activation = 'relu'

    model_input = Input(shape=(input_shape,), dtype='float32')

    net = Dense(32, activation=activation, name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)
    net = Dense(32, activation=activation, name='dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = Dense(32, activation=activation, name='dense3',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = Dense(32, activation=activation, name='dense4',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    preds = Dense(10, activation='softmax', name='dense5',
                  kernel_regularizer=regularizers.l2(1e-3))(net)
    model = Model(model_input, preds)
    model.load_weights('models/' + datatype + '_blackbox_extra.hdf5',
                       by_name=True)
    pred_model = Model(model_input, preds)
    pred_model.compile(loss=None,
                       optimizer='rmsprop',
                       metrics=None)
    if calculate:
        total_lipschitz[1, :] = calculate_prob_lipschitz(x_val, pred_model,
                                                   r=median_rad,
                                                   L_range=L_range,
                                                   num_points=len(x_val))

    del pred_model
    print('Training Linear Classifier')

    activation = None

    model_input = Input(shape=(input_shape,), dtype='float32')

    net = Dense(32, activation=activation, name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)

    preds = Dense(10, activation='softmax', name='dense3',
                  kernel_regularizer=regularizers.l2(1e-3))(net)
    model = Model(model_input, preds)

    model.load_weights('models/' + datatype + '_blackbox_linear.hdf5',
                       by_name=True)
    pred_model = Model(model_input, preds)
    pred_model.compile(loss=None,
                       optimizer='rmsprop',
                       metrics=None)


    if calculate:
        total_lipschitz[2, :] = calculate_prob_lipschitz(x_val, pred_model,
                                                   r=median_rad,
                                                   L_range=L_range,
                                                   num_points=len(x_val))

    del pred_model
    ###


    print("SVM")
    svm_classif = pickle.load(open('models/' + datatype + '_svm.pk', 'rb'))

    if calculate:
        total_lipschitz[3, :] = calculate_prob_lipschitz(x_val, svm_classif,
                                                   r=median_rad,
                                                   L_range=L_range,
                                                   num_points=len(x_val),
                                                   NN=False)


    if calculate:
        pickle.dump(total_lipschitz, open(save_lipschitz, 'wb'))
    else:
        total_lipschitz = pickle.load(open(save_lipschitz, 'rb'))

    image_name = 'plots/classifiers_' + datatype + '_lipschitz.PNG'
    plt.figure()
    for i in range(len(classifiers)):
        plt.errorbar(x=L_range, y=total_lipschitz[i, :], yerr=0,
                     label=classifiers[i], marker='x')
    plt.legend()
    plt.savefig(image_name)
    plt.show()
    plt.close()


# In[ ]:




