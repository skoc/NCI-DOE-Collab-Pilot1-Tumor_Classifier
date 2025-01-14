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

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import tc1 as bmk
import candle

from utility import *

def eprint(args):
    sys.stderr.write(str(args) + "\n")

def initialize_parameters(default_model = 'tc1_default_model.json'):

    # Build benchmark object
    tc1Bmk = bmk.BenchmarkTC1(file_path, default_model, 'keras',
    prog='tc1_baseline', desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.finalize_parameters(tc1Bmk, format_config="json")
    eprint(f"[DEBUG] gParameters: {gParameters}")
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def run(gParameters, train_data, test_data, default_model = 'tc1_default_model.json'):

    X_train, Y_train, X_test, Y_test = bmk.load_data(gParameters, train_data, test_data)

    # Filter args for better output naming
    lst_row = ['model_name', 'epochs', 'dense', 'out_activation', \
               'optimizer', 'activation', 'metrics', 'dropout', 'conv', \
               'batch_size', 'loss', 'pool']
    dict_row = {k:v for k, v in gParameters.items() if k in lst_row}
    name_unique_ext = ':'.join(map(str, dict_row.values())).replace('[', '').replace(']', '').replace(' ', '').replace(':', '_').replace(',', '-')

    eprint(f'X_train shape: {X_train.shape}')
    eprint(f'X_test shape: {X_test.shape}')

    eprint(f'Y_train shape: {Y_train.shape}')
    eprint(f'Y_test shape: {Y_test.shape}')

    x_train_len = X_train.shape[1]

    # this reshaping is critical for the Conv1D to work

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    eprint(f'X_train shape: {X_train.shape}')
    eprint(f'X_test shape: {X_test.shape}')

    model = Sequential()
    dense_first = True

    layer_list = list(range(0, len(gParameters['conv']), 3))
    for l, i in enumerate(layer_list):
        filters = gParameters['conv'][i]
        filter_len = gParameters['conv'][i+1]
        stride = gParameters['conv'][i+2]
        print(i/3, filters, filter_len, stride)
        if gParameters['pool']:
            pool_list=gParameters['pool']
            if type(pool_list) != list:
                pool_list=list(pool_list)

        if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
        dense_first = False
        if 'locally_connected' in gParameters:
                model.add(LocallyConnected1D(filters, filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
        else:
            #input layer
            if i == 0:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid', input_shape=(x_train_len, 1)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=filter_len, strides=stride, padding='valid'))
        model.add(Activation(gParameters['activation']))
        if gParameters['pool']:
                model.add(MaxPooling1D(pool_size=pool_list[i//3]))

    if not dense_first:
        model.add(Flatten())

    for i, layer in enumerate(gParameters['dense']):
        if layer:
            if i == 0 and dense_first:
                model.add(Dense(layer, input_shape=(x_train_len, 1)))
            else:
                model.add(Dense(layer))
            model.add(Activation(gParameters['activation']))
            if gParameters['dropout']:
                    model.add(Dropout(gParameters['dropout']))

    if dense_first:
        model.add(Flatten())

    model.add(Dense(gParameters['classes']))
    model.add(Activation(gParameters['out_activation']))

    model.summary()
    eprint(model.summary())

    model.compile(loss=gParameters['loss'],
              optimizer=gParameters['optimizer'],
              metrics=[gParameters['metrics']])

    output_dir = gParameters['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # set up callbacks to do work during model training..
    model_name = gParameters['model_name']
    path = '{}/trained_model_best_{}.autosave.model.h5'.format(output_dir, name_unique_ext)
    checkpointer = ModelCheckpoint(filepath=path, verbose=1, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger('{}/log_{}.training.csv'.format(output_dir, name_unique_ext))
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    history = model.fit(X_train, Y_train,
                    batch_size=gParameters['batch_size'],
                    epochs=gParameters['epochs'],
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks = [checkpointer, csv_logger, reduce_lr])

    # Print History in Log Screen
    print_history(history, validation = True)

    score = model.evaluate(X_test, Y_test, verbose=0)

    eprint(f'[DEBUG] Test score: {score[0]}')
    eprint(f'[DEBUG] Test accuracy: {score[1]}')

    preds_test = model.predict(X_test)

    result_test_class = [np.argmax(y, axis=None, out=None) for y in preds_test]
    eprint(f'[DEBUG] result_test_class: {result_test_class}')

    Y_test_class = [np.argmax(y, axis=None, out=None) for y in Y_test]

    ###################################################

    # Decode Class Labels - Disease Types
    df_disease = pd.read_table('type_18_class_labels', header= None, names = ['index', 'name'])
    lst_names = [df_disease.loc[df_disease["index"]==idx, "name"].values[0] for idx in list(sorted(set(Y_test_class)))]
    
    # Plot AUC
    eprint('[DEBUG] Stats [1] Calculating plot_auc_multi_class...')
    plot_auc_multi_class(Y_test_class, result_test_class, list(set(Y_test_class)), parameters_dict=dict_row, title='', figsize=(12,12), lst_disease=lst_names)

    # Plot Loss
    eprint('[DEBUG] Stats [2] Calculating plot_loss...')
    plot_loss(history, parameters_dict=dict_row, title='')

    # Plot Prediction
    eprint('[DEBUG] Stats [3] Calculating plot_predictions...')
    plot_predictions(Y_test_class, result_test_class, parameters_dict = dict_row, title='')

    # Calculate Metrics
    eprint('[DEBUG] Stats [4] Calculating calculate_metrics...')
    d = {}
    d['Accuracy'], d['Precision'], d['Recall'], d['F-Score'] = calculate_metrics(Y_test_class, result_test_class, average="macro")
    
    # Plot Confusion Matrix
    eprint('[DEBUG] Stats [5] Calculating confusion_matrix...')
    cm_val = confusion_matrix(Y_test_class, result_test_class)
    eprint(f'[DEBUG] cm_val: {cm_val}')
    # d['TN Rate'], d['FP Rate'] = plot_confusion_matrix(cm_val, list(sorted(set(Y_test_class))), parameters_dict = {}, title=model_name)
    d['TN Rate'], d['FP Rate'] = plot_confusion_matrix(cm_val, lst_names, parameters_dict = dict_row, title='', figsize=(20,20))

    # Generate Report
    eprint('[DEBUG] Stats [6] Generating report...')

    conf_col = ':'.join(list(dict_row.keys())) # skip some keys for better visual
    columns = [conf_col] + list(d.keys())
    df_out = pd.DataFrame(columns=columns)
    df_out.loc[0] = [':'.join(map(str, list(dict_row.values())))] + list(d.values())
    name_report = 'report_' + model_name + '_' + default_model.split('/')[-1].split('.')[0] + '.csv'
    df_out.to_csv(name_report, index=False)

    eprint('[DEBUG] Stats Done!')

    ###################################################
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("{}/trained_model{}.model.json".format(output_dir, name_unique_ext), "w") as json_file:
        json_file.write(model_json)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("{}/trained_model{}.model.yaml".format(output_dir, name_unique_ext), "w") as yaml_file:
        yaml_file.write(model_yaml)


    # serialize weights to HDF5
    model.save_weights("{}/trained_model{}.model.h5".format(output_dir, name_unique_ext))
    eprint("[DEBUG] Saved model to disk")

    # load json and create model
    json_file = open('{}/trained_model{}.model.json'.format(output_dir, name_unique_ext), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)


    # load yaml and create model
    yaml_file = open('{}/trained_model{}.model.yaml'.format(output_dir, name_unique_ext), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model_yaml = model_from_yaml(loaded_model_yaml)


    # load weights into new model
    loaded_model_json.load_weights('{}/trained_model{}.model.h5'.format(output_dir, name_unique_ext))
    eprint("[DEBUG] Loaded json model from disk")

    # evaluate json loaded model on test data
    loaded_model_json.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

    eprint(f'[DEBUG] json Test score: {score_json[0]}')
    eprint(f'[DEBUG] json Test accuracy: {score_json[1]}')

    eprint(f"[DEBUG] json {loaded_model_json.metrics_names[1]}: {score_json[1]*100}")


    # load weights into new model
    loaded_model_yaml.load_weights('{}/trained_model{}.model.h5'.format(output_dir, name_unique_ext))
    eprint(f"[DEBUG] Loaded yaml model from disk")

    # evaluate loaded model on test data
    loaded_model_yaml.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])
    score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

    eprint(f'[DEBUG] yaml Test score: {score_yaml[0]}')
    eprint(f'[DEBUG] yaml Test accuracy: {score_yaml[1]}')

    eprint(f"[DEBUG] yaml {loaded_model_yaml.metrics_names[1]}: {score_yaml[1]*100}")

    return history


def main(train_data, test_data, config_file):

    gParameters = initialize_parameters(config_file)
    run(gParameters, train_data, test_data, config_file)

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', help="Train Data from the Platform", required=True)
    parser.add_argument('--test_data', help="Test Data from the Platform", required=True)
    parser.add_argument('--config_file', help="Parameters of the model", default='tc1_default_model.json')

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    config_file = args.config_file

    main(train_data, test_data, config_file)

    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass

