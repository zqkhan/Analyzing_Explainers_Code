import numpy as np
import pickle
from datetime import datetime
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras import regularizers
from keras.utils import to_categorical
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.svm import SVC
np.random.seed(0)

#for datatype in ['orange_skin', 'XOR', 'nonlinear_additive', 'switch']:
for datatype in ['cifar1006']:
    train = True
    BATCH_SIZE = 32
    epochs = 30


    cifar = pickle.load(open('data/cifar10_6_6_train.pkl', 'rb'))
    data = cifar[1]
    labels = cifar[0]
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=42)
    x_train, x_val = np.array(x_train), np.array(x_val)
    x_train, x_val = x_train.reshape((len(x_train), -1)), x_val.reshape((len(x_val), -1))
    y_train_orig, y_val_orig = y_train.copy(), y_val.copy()
    y_train, y_val = to_categorical(y_train), to_categorical(y_val)
    input_shape = x_train.shape[-1]
    activation = 'relu'

    model_input = Input(shape=(input_shape,), dtype='float32')

    net = Dense(32, activation=activation, name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)

    preds = Dense(10, activation='softmax', name='dense3',
                  kernel_regularizer=regularizers.l2(1e-3))(net)
    model = Model(model_input, preds)

    if train:
        adam = optimizers.Adam(lr=1e-3)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['acc'])
        filepath = 'models/' + datatype + '_blackbox.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=callbacks_list, epochs=epochs,
                  batch_size=BATCH_SIZE)
        st2 = time.time()
    else:
        model.load_weights('models/' + datatype + '_blackbox.hdf5',
                           by_name=True)
    pred_model = Model(model_input, preds)
    pred_model.compile(loss=None,
                       optimizer='rmsprop',
                       metrics=None)
    pred_val = pred_model.predict(x_val, verbose=1, batch_size=BATCH_SIZE)
    del pred_model
    ######
    print('Training Linear Classifier')

    activation = None

    model_input = Input(shape=(input_shape,), dtype='float32')

    net = Dense(32, activation=activation, name='dense1',
                kernel_regularizer=regularizers.l2(1e-3))(model_input)

    preds = Dense(10, activation='softmax', name='dense3',
                  kernel_regularizer=regularizers.l2(1e-3))(net)
    model = Model(model_input, preds)

    if train:
        adam = optimizers.Adam(lr=1e-3)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['acc'])
        filepath = 'models/' + datatype + '_blackbox_linear.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=callbacks_list, epochs=epochs,
                  batch_size=BATCH_SIZE)
        st2 = time.time()
    else:
        model.load_weights('models/' + datatype + '_blackbox_linear.hdf5',
                           by_name=True)
    pred_model = Model(model_input, preds)
    pred_model.compile(loss=None,
                       optimizer='rmsprop',
                       metrics=None)
    pred_val = pred_model.predict(x_val, verbose=1, batch_size=BATCH_SIZE)

    ###
    del pred_model
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

    if train:
        adam = optimizers.Adam(lr=1e-3)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['acc'])
        filepath = 'models/' + datatype + '_blackbox_extra.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=callbacks_list, epochs=epochs,
                  batch_size=BATCH_SIZE)
        st2 = time.time()
    else:
        model.load_weights('models/' + datatype + '_blackbox_extra.hdf5',
                           by_name=True)
    pred_model = Model(model_input, preds)
    pred_model.compile(loss=None,
                       optimizer='rmsprop',
                       metrics=None)
    pred_val = pred_model.predict(x_val, verbose=1, batch_size=BATCH_SIZE)

    ### train SVM

    print("train svm")
    if train:
        svm_classif = SVC(probability=True).fit(x_train, y_train_orig.astype(int))
        pickle.dump(svm_classif,file=open('models/' + datatype + '_svm.pk', 'wb'))
    else:
        svm_classif = pickle.load(open('models/' + datatype + '_svm.pk', 'rb'))

    pred_val = svm_classif.predict_proba(x_val)

    r = 3