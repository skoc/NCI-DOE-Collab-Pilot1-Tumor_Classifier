from __future__ import print_function

import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import tc1 as bmk
import candle


def initialize_parameters(default_model = 'tc1_default_model.txt'):

    # Build benchmark object
    tc1Bmk = bmk.BenchmarkTC1(file_path, default_model, 'keras',
    prog='tc1_baseline', desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.finalize_parameters(tc1Bmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def run(gParameters):


    # load json and create model
    trained_model_json = gParameters['trained_model_json']
    json_data_url =  gParameters['data_url']  + trained_model_json 
    candle.get_file(trained_model_json, json_data_url, datadir=".")

    json_file = open(trained_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load weights into new model
    trained_model_h5 = gParameters['trained_model_h5']
    h5_data_url =  gParameters['data_url']  + trained_model_h5
    candle.get_file(trained_model_h5, h5_data_url, datadir=".")
    loaded_model_json.load_weights(trained_model_h5)

    loaded_model_json.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])

    # evaluate json loaded model on test data
    X_train, Y_train, X_test, Y_test = bmk.load_data(gParameters)

    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    # this reshaping is critical for the Conv1D to work
    X_test = np.expand_dims(X_test, axis=2)

    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)
    print('json Test score:', score_json[0])
    print('json Test accuracy:', score_json[1])
    print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))

def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass

