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
from keras.utils import to_categorical

from tensorflow.python.keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from utils.explanations import calculate_robust_astute_sampled

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


datatype = 'cifar1006'
run_times = 20
prop_points = 0.05
calculate = True
#epsilon_range = np.arange(0.01, 3.1, 0.05)
epsilon_range = [0.31]

# In[5]:


cifar = pickle.load(open('data/cifar10_6_6_train.pkl', 'rb'))
data = cifar[1]
labels = cifar[0]
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=42)
x_train, x_val = np.array(x_train), np.array(x_val)
x_train, x_val = x_train.reshape((len(x_train), -1)), x_val.reshape((len(x_val), -1))
y_train, y_val = to_categorical(y_train), to_categorical(y_val)
input_shape = x_train.shape[-1]


# In[6]:


median_rad = np.median(pdist(x_train))


# In[10]:


save_astuteness_file = 'plots/rise_' + datatype + '_astuteness_classifiers_lip_single_epsilon.pk'
lambda_dense_list = [float(1.5), float(2), float("inf")]
lambda_names = ['Regularized High', 'Regularized Low', 'Not Regularized']


# In[8]:


if calculate:
    total_astuteness = np.zeros(shape=(run_times, len(lambda_names), len(epsilon_range)))
    for i in range(run_times):
        print('Completing Run ' + str(i + 1) + ' of ' + str(run_times))
        for (j, lambda_dense) in enumerate(lambda_dense_list):
            all_layer_weights = pickle.load(open('extracted_weights/cifar1006_l2_' + str(j) + '.pk', 'rb'))
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
            fname = 'explained_weights/rise/' + 'rise_' + datatype + '_' + str(j) + '_' + str(i) + '_lip.gz'
            explanations = np.loadtxt(fname, delimiter=',')
            for k in tqdm(range(len(epsilon_range))):
                _, total_astuteness[i, j, k], _ = calculate_robust_astute_sampled(data=x_val,
                                                                                  explainer=pred_model,
                                                                                  explainer_type='rise',
                                                                                  explanation_type='attribution',
                                                                                  ball_r=median_rad,
                                                                                  epsilon=epsilon_range[k],
                                                                                  num_points=int(
                                                                                      prop_points * len(
                                                                                          x_val)),
                                                                                  NN=True,
                                                                                  data_explanation=explanations)
    pickle.dump(total_astuteness, open(save_astuteness_file, 'wb'))
else:
    total_astuteness = pickle.load(open(save_astuteness_file, 'rb'))


# In[9]:


astuteness_mean = total_astuteness.mean(axis=0)
astuteness_std = total_astuteness.std(axis=0)
image_name = 'plots/rise_' + datatype + '_astuteness_classifiers.PNG'
fig, ax = plt.subplots()
for i in range(len(lambda_names)):
    ax.errorbar(x=epsilon_range, y=astuteness_mean[i, :], yerr=astuteness_std[i, :],
                label=lambda_names[i])
plt.legend()
plt.savefig(image_name)
plt.show()
plt.close()


# In[11]:


pickle.dump(total_astuteness, open(save_astuteness_file, 'wb'))


# In[ ]:




