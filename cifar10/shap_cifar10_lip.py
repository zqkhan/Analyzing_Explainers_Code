#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import pickle
import numpy as np
import argparse

import mat73
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model, Sequential
from keras.utils import to_categorical
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from utils.explanations import calculate_robust_astute_sampled
import shap

np.random.seed(0)


# In[3]:


def set_all_weights(model, all_layer_weights):
    count = 0
    for layer in model.layers:
        if type(layer) is Dense:
            count += 1
    if count == len(all_layer_weights):
        c = 0
        for layer in model.layers:
            if type(layer) is Dense:
                layer.set_weights(all_layer_weights[c])
                c += 1
        return model
    else:
        print("models don't match")


# In[4]:


def shap_explainer(datatype, ball_r, epsilon, prop_points, exponentiate, lambda_names, all_layer_weights):
    blackbox_path = 'models/' + datatype + '_blackbox.hdf5'
    cifar = pickle.load(open('data/cifar10_6_6_train.pkl', 'rb'))
    data = cifar[1]
    labels = cifar[0]
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=42)
    x_train, x_val = np.array(x_train), np.array(x_val)
    x_train, x_val = x_train.reshape((len(x_train), -1)), x_val.reshape((len(x_val), -1))
    y_train, y_val = to_categorical(y_train), to_categorical(y_val)
    input_shape = x_train.shape[-1]

    activation = 'relu'

    model_input = Input(shape=(input_shape,), dtype='float32')

    net = Dense(64, activation=activation, name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)
    net = Dense(64, activation=activation, name='dense2',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = Dense(64, activation=activation, name='dense3',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    net = Dense(64, activation=activation, name='dense4',
                kernel_regularizer=regularizers.l2(1e-3))(net)
    preds = Dense(10, activation='softmax', name='dense5',
                  kernel_regularizer=regularizers.l2(1e-3))(net)
    bbox_model = Model(model_input, preds)
    bbox_model = set_all_weights(bbox_model, all_layer_weights)
    pred_model = Model(model_input, preds)

    background = x_train[np.random.choice(len(x_train), 100, replace=False)]
    explainer = shap.GradientExplainer(bbox_model, background)

    explanation = calculate_robust_astute_sampled(data=x_val,
                                                  explainer=explainer,
                                                  explainer_type='shap',
                                                  explanation_type='attribution',
                                                  ball_r=ball_r,
                                                  epsilon=epsilon,
                                                  num_points=int(prop_points * len(x_val)),
                                                  exponentiate=exponentiate,
                                                  calculate_astuteness=False)

    del pred_model
    return np.abs(explanation)


# In[5]:


ball_radius = 2
epsilon = 0.05
prop_points = 0.05
run_times = 20
exponentiate = 0
classifiers = ['4layer']
lambda_dense_list =[float(1.25), float(1.5), float("inf")]
lambda_names = ['Regularized High', 'Regularized Low', 'Not Regularized']
for datatype in ['cifar1006']:
    for c in range(len(lambda_names)):
        print('Calculating for: ' + lambda_names[c])
        for i in range(run_times):
            print(i)
            fname = 'explained_weights/shap/' + 'shap_' + datatype + '_' + str(c) + '_' + str(i) + '_lip.gz'
            all_layer_weights = pickle.load(open('extracted_weights/cifar1006_l2_' + str(c) + '.pk', 'rb'))

            explanation = shap_explainer(datatype=datatype,
                                           ball_r=ball_radius,
                                           epsilon=epsilon,
                                           prop_points=prop_points,
                                           exponentiate=exponentiate,
                                           lambda_names=lambda_names[c],
                                           all_layer_weights=all_layer_weights)
            np.savetxt(X=explanation, fname=fname, delimiter=',')


# In[ ]:




