#!/usr/bin/env python

import numpy as np
from scipy.io import savemat
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint
from arch.maybe import maybe_batchnorm, maybe_dropout
from arch.lipschitz import lcc_conv, lcc_dense, SpectralDecay
import getopt
import os
from sys import argv
import pandas as pd
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)


def extract_weights(model):
    all_weights = model.get_weights()
    relevant_weights = [a.T.astype(np.double) for (i, a) in enumerate(all_weights) if i % 2 == 0]
    data = {'weights': np.array(relevant_weights, dtype=np.object)}

    return data


def extract_all_weights(model):
    all_layer_weights = []
    for layer in model.layers:
        if type(layer) is Dense:
            layer_weights = layer.get_weights()
            all_layer_weights.append(layer_weights)
    return all_layer_weights


batch_size = 32
epochs = 100
num_classes = 10
lcc_norm = 2
lambda_conv = float("inf")
lambda_dense_list = [float(1.5), float(2.5), float("inf")]
lambda_bn = float("inf")
drop_conv = 0
drop_dense = 0
sd_conv = 0
sd_dense = 0
batchnorm = False
model_path = ""
valid = False
img_rows, img_cols = 28, 28
loaded = False
log_path = ""
arch = "mlp"

train = True
datatype = 'mnist'

opts, args = getopt.getopt(argv[1:], "", longopts=[
    "dataset=",
    "valid",
    "lcc=",
    "lambda-conv=",
    "lambda-dense=",
    "lambda-bn=",
    "drop-conv=",
    "drop-dense=",
    "sd-conv=",
    "sd-dense=",
    "batchnorm",
    "model-path=",
    "log-path=",
    "arch="
])

for (k, v) in opts:
    if k == "--dataset":
        loaded = True
        if v == "mnist":
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif v == "fashion-mnist":
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            raise Exception("Unknown dataset")
    elif k == "--valid":
        valid = True
    elif k == "--lcc":
        lcc_norm = float(v)
    elif k == "--lambda-conv":
        lambda_conv = float(v)
    elif k == "--lambda-dense":
        lambda_dense = float(v)
    elif k == "--lambda-bn":
        lambda_bn = float(v)
    elif k == "--drop-conv":
        drop_conv = float(v)
    elif k == "--drop-dense":
        drop_dense = float(v)
    elif k == "--sd-conv":
        sd_conv = float(v)
    elif k == "--sd-dense":
        sd_dense = float(v)
    elif k == "--batchnorm":
        batchnorm = True
    elif k == "--model-path":
        model_path = v
    elif k == "--log-path":
        log_path = v
    elif k == "--arch":
        arch = v

if not loaded:
    mnist = pickle.load(open('data/mnist_10_10_train.pkl', 'rb'))
    data = mnist[1]
    labels = mnist[0]
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=42)
    x_train = np.array(x_train).reshape(-1, x_train[0].shape[0]*x_train[0].shape[1])
    x_val = np.array(x_val).reshape(-1, x_val[0].shape[0]*x_val[0].shape[1])
    y_train, y_val = np.array(y_train), np.array(y_val)
    y_train, y_val = to_categorical(y_train), to_categorical(y_val)
    input_shape = x_train.shape[-1]


# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    if epoch >= 20:
        return 0.00001
    else:
        return 0.0001


for (i, lambda_dense) in enumerate(lambda_dense_list):

    lr_scheduler = LearningRateScheduler(lr_schedule)
    opt = adam(amsgrad=True)

    conv_reg = SpectralDecay(sd_conv)
    dense_reg = SpectralDecay(sd_dense)

    model = Sequential()
    model.add(Dense(32, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
    # maybe_batchnorm(model, lambda_bn, batchnorm)
    model.add(Activation("relu"))
    model.add(Dense(32, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
    # maybe_batchnorm(model, lambda_bn, batchnorm)
    model.add(Activation("relu"))
    model.add(Dense(32, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
    # maybe_batchnorm(model, lambda_bn, batchnorm)
    model.add(Activation("relu"))
    model.add(Dense(32, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
    # maybe_batchnorm(model, lambda_bn, batchnorm)
    model.add(Activation("relu"))
    model.add(Dense(num_classes, kernel_regularizer=dense_reg, **lcc_dense(lcc_norm, lambda_dense)))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    if train:

        filepath = 'models/' + datatype + '_blackbox_' + str(i) + '.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_val, y_val),
                  callbacks=callbacks_list)
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  verbose=1,
                  validation_data=(x_val, y_val), )
        model.load_weights('models/' + datatype + '_blackbox_' + str(i) + '.hdf5', by_name=True)

    all_layer_weights = extract_all_weights(model)
    pickle.dump(all_layer_weights, open('extracted_weights/mnist_l2_' + str(i) + '.pk', 'wb'))
    data = extract_weights(model)
    savemat('extracted_weights/mnist_l2_' + str(i) + '.mat', data)

score = model.evaluate(x_test, y_test, verbose=0)

# with open(log_path, "a") as f:
#     f.write("loss=" + str(score[0]) + ",accuracy=" + str(score[1]) + "\n")
