#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist

import cxplain
from cxplain import MLPModelBuilder, ZeroMasking, CXPlain
from tensorflow.python.keras.losses import binary_crossentropy, categorical_crossentropy

from tensorflow.python.keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from utils.explanations import calculate_robust_astute_sampled
import shap

np.random.seed(0)


# In[3]:


datatype = 'mnist'
run_times = 5
prop_points = 1
calculate = True
epsilon_range = np.arange(0.01, 1.1, 0.05)
masking_operation = ZeroMasking()
loss = binary_crossentropy


# In[4]:


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


# In[6]:


save_astuteness_file = 'plots/cxplain_' + datatype + '_astuteness_classifiers.pk'
classifiers = ['2layer', '4layer', 'linear', 'svm']


# In[7]:


if calculate:
    median_rad = np.median(pdist(x_train))
    total_astuteness = np.zeros(shape=(run_times, len(classifiers), len(epsilon_range)))
    for i in range(run_times):
        print('Completing Run ' + str(i + 1) + ' of ' + str(run_times))
        for j in range(len(classifiers)):
            if classifiers[j] == '2layer':
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
                
                model_builder = MLPModelBuilder(num_layers=1, num_units=32, activation=activation, verbose=1,
                        batch_size=32, learning_rate=0.001, num_epochs=5, early_stopping_patience=15,
                        with_bn=False)


            elif classifiers[j] == '4layer':
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
                model_builder = MLPModelBuilder(num_layers=4, num_units=32, activation=activation, verbose=1,
                                batch_size=32, learning_rate=0.001, num_epochs=5, early_stopping_patience=15,
                                with_bn=False)


            elif classifiers[j] == 'linear':
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
                model_builder = MLPModelBuilder(num_layers=1, num_units=32, activation=activation, verbose=1,
                                        batch_size=32, learning_rate=0.001, num_epochs=5, early_stopping_patience=15,
                                        with_bn=False)
            elif classifiers[j] == 'svm':
                pred_model = pickle.load(open('models/' + datatype + '_svm.pk', 'rb'))
                model_builder = MLPModelBuilder(num_layers=2, num_units=32, activation=activation, verbose=1,
                                                batch_size=32, learning_rate=0.001, num_epochs=5, 
                                                early_stopping_patience=15,
                                                with_bn=False)
            if classifiers[j] == 'svm':
                explainer = CXPlain(pred_model, model_builder, masking_operation, loss, num_models=1)
                explainer.fit(x_train, y_train)
                for k in tqdm(range(len(epsilon_range))):
                    _, total_astuteness[i, j, k], _ = calculate_robust_astute_sampled(data=x_val,
                                                                                      explainer=explainer,
                                                                                      explainer_type='cxplain',
                                                                                      explanation_type='attribution',
                                                                                      ball_r=median_rad,
                                                                                      epsilon=epsilon_range[k],
                                                                                      num_points=int(
                                                                                          prop_points * len(
                                                                                              x_val)),
                                                                                      NN=False,
                                                                                      data_explanation=None)
            else:
                explainer = CXPlain(pred_model, model_builder, masking_operation, loss, num_models=1)
                explainer.fit(x_train, y_train)
                for k in tqdm(range(len(epsilon_range))):
                    _, total_astuteness[i, j, k], _ = calculate_robust_astute_sampled(data=x_val,
                                                                                      explainer=explainer,
                                                                                      explainer_type='cxplain',
                                                                                      explanation_type='attribution',
                                                                                      ball_r=median_rad,
                                                                                      epsilon=epsilon_range[k],
                                                                                      num_points=int(
                                                                                          prop_points * len(
                                                                                              x_val)),
                                                                                      NN=True,
                                                                                      data_explanation=None)
    pickle.dump(total_astuteness, open(save_astuteness_file, 'wb'))
else:
    total_astuteness = pickle.load(open(save_astuteness_file, 'rb'))
astuteness_mean = total_astuteness.mean(axis=0)
astuteness_std = total_astuteness.std(axis=0)
image_name = 'plots/cxplain_' + datatype + '_astuteness_classifiers.PNG'
fig, ax = plt.subplots()
for i in range(len(classifiers)):
    ax.errorbar(x=epsilon_range, y=astuteness_mean[i, :], yerr=astuteness_std[i, :],
                label=classifiers[i])
plt.legend()
plt.savefig(image_name)
plt.show()
plt.close()


# In[8]:


x_train.shape


# In[9]:


x_val.shape


# In[ ]:




