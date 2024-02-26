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
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from keras.utils import to_categorical


from utils.explanations import calculate_robust_astute_sampled
import shap

np.random.seed(0)


# In[3]:


def shap_explainer(datatype, ball_r, epsilon, prop_points, exponentiate, classifier):
    blackbox_path = 'models/' + datatype + '_blackbox.hdf5'
    mnist = pickle.load(open('data/mnist_10_10_train.pkl', 'rb'))
    data = mnist[1]
    labels = mnist[0]
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=42)
    x_train = np.array(x_train).reshape(-1, x_train[0].shape[0]*x_train[0].shape[1])
    x_val = np.array(x_val).reshape(-1, x_val[0].shape[0]*x_val[0].shape[1])
    y_train, y_val = np.array(y_train), np.array(y_val)
    y_train_orig, y_val_orig = y_train.copy(), y_val.copy()
    y_train, y_val = to_categorical(y_train), to_categorical(y_val)
    input_shape = x_train.shape[-1]
    
    if classifier == '2layer':
        activation = 'relu'

        model_input = Input(shape=(input_shape,), dtype='float32')

        net = Dense(32, activation=activation, name='dense1',
                    kernel_regularizer=regularizers.l2(1e-3))(model_input)

        preds = Dense(10, activation='softmax', name='dense3',
                      kernel_regularizer=regularizers.l2(1e-3))(net)
        bbox_model = Model(model_input, preds)

        bbox_model.load_weights('models/' + datatype + '_blackbox.hdf5',
                                by_name=True)
        pred_model = Model(model_input, preds)

    elif classifier == '4layer':
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
        bbox_model = Model(model_input, preds)
        bbox_model.load_weights('models/' + datatype + '_blackbox_extra.hdf5',
                                by_name=True)
        pred_model = Model(model_input, preds)


    elif classifier == 'linear':
        activation = None

        model_input = Input(shape=(input_shape,), dtype='float32')

        net = Dense(32, activation=activation, name='dense1',
                    kernel_regularizer=regularizers.l2(1e-3))(model_input)

        preds = Dense(10, activation='softmax', name='dense3',
                      kernel_regularizer=regularizers.l2(1e-3))(net)
        bbox_model = Model(model_input, preds)
        bbox_model.load_weights('models/' + datatype + '_blackbox_linear.hdf5',
                                by_name=True)
        pred_model = Model(model_input, preds)


    elif classifier == 'svm':
        pred_model = pickle.load(open('models/' + datatype + '_svm.pk', 'rb'))

    if classifier == 'svm':
#         training_indices = np.random.choice(len(x_train), int(0.001*len(x_train)), replace=False)
        explainer = shap.KernelExplainer(pred_model.predict_proba, shap.kmeans(x_train, 100))


        explanation = calculate_robust_astute_sampled(data=x_val,
                                                   explainer=explainer,
                                                   explainer_type='shap',
                                                   explanation_type='attribution',
                                                   ball_r=ball_r,
                                                   epsilon=epsilon,
                                                   num_points=int(prop_points * len(x_val)),
                                                   exponentiate=exponentiate,
                                                   calculate_astuteness=False,
                                                   NN=False)
    else:
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


# In[4]:


ball_radius = 2
epsilon = 0.05
prop_points = 0.05
run_times = 5
exponentiate = 0
classifiers = ['2layer', 'linear', '4layer', 'svm']

for datatype in ['mnist']:
    for c in range(len(classifiers)):
        for i in range(run_times):
            fname = 'explained_weights/shap/' + 'shap_' + datatype + '_' + classifiers[c] + '_' + str(i) + '.gz'
            explanation = shap_explainer(datatype=datatype,
                                           ball_r=ball_radius,
                                           epsilon=epsilon,
                                           prop_points=prop_points,
                                           exponentiate=exponentiate,
                                           classifier=classifiers[c])
            np.savetxt(X=explanation, fname=fname, delimiter=',')


# In[ ]:




