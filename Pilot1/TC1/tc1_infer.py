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

from utility import *

def eprint(args):
    sys.stderr.write(str(args) + "\n")

def initialize_parameters(default_model = 'tc1_default_model.txt'):

    # Build benchmark object
    tc1Bmk = bmk.BenchmarkTC1(file_path, default_model, 'keras',
    prog='tc1_baseline', desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')

    # Initialize parameters
    gParameters = candle.finalize_parameters(tc1Bmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters


def run(gParameters, trained_model_json, trained_model_h5, test_data):


    # load json and create model
    # trained_model_json = gParameters['trained_model_json']
    # json_data_url =  gParameters['data_url']  + trained_model_json 
    # candle.get_file(trained_model_json, json_data_url, datadir=".")

    json_file = open(trained_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load weights into new model
    # trained_model_h5 = gParameters['trained_model_h5']
    # h5_data_url =  gParameters['data_url']  + trained_model_h5
    # candle.get_file(trained_model_h5, h5_data_url, datadir=".")
    loaded_model_json.load_weights(trained_model_h5)

    loaded_model_json.compile(loss=gParameters['loss'],
            optimizer=gParameters['optimizer'],
            metrics=[gParameters['metrics']])

    # evaluate json loaded model on test data
    X_test, Y_test = bmk.load_test_data(gParameters, test_data)

    eprint(f"[DEBUG] X_test shape: {X_test.shape}")
    eprint(f"[DEBUG] Y_test shape: {Y_test.shape}")

    # this reshaping is critical for the Conv1D to work
    X_test = np.expand_dims(X_test, axis=2)

    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)
    eprint(f"[DEBUG] json Test score: {score_json[0]}")
    eprint(f"[DEBUG] json Test accuracy: {score_json[1]}" )
    eprint(f"[DEBUG] json {loaded_model_json.metrics_names[1]}: {score_json[1]*100}")
    eprint(f"[DEBUG] score_json: {score_json}")

    # Write Test Scores to File
    dict_output = {}
    dict_output['Test score'], dict_output['Test accuracy']= score_json[0], score_json[1]
    
    columns = list(dict_output.keys())
    df_out = pd.DataFrame(columns=columns)
    df_out.loc[0] = list(dict_output.values())
    df_out.to_csv(gParameters['model_name'] + '.csv', index=False)

    prepare_detailed_outputs(loaded_model_json,X_test, Y_test)


def prepare_detailed_outputs(model,X_test, Y_test):
    dict_row={"test":"test"}
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
    # eprint('[DEBUG] Stats [2] Calculating plot_loss...')
    # plot_loss(history, parameters_dict=dict_row, title='')

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
    name_report = 'report_testing.csv'
    df_out.to_csv(name_report, index=False)

    eprint('[DEBUG] Stats Done!')


def main(trained_model_json, trained_model_h5, test_data, tc1_default_model):

    gParameters = initialize_parameters(default_model = tc1_default_model)
    eprint(f"[DEBUG] gParameters: {gParameters}")
    run(gParameters, trained_model_json, trained_model_h5, test_data)

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_json', help="Json file of Trained Model", required=True)
    parser.add_argument('--trained_model_h5', help="Weights of Trained Model", required=True)
    # parser.add_argument('--train_data', help="Train Data from the Platform", required=True)
    parser.add_argument('--test_data', help="Test Data from the Platform", required=True)
    parser.add_argument('--config_file', help="Parameters of the model", required=True)

    args = parser.parse_args()

    trained_model_json = args.trained_model_json
    trained_model_h5 = args.trained_model_h5
    # train_data = args.train_data
    test_data = args.test_data
    tc1_default_model = args.config_file

    # Path fix for empty tc1_default_model inputs, when it's copied from the required files section
    # tc1_default_model = os.path.dirname(test_data) + os.path.basename(tc1_default_model)

    main(trained_model_json=trained_model_json, 
        trained_model_h5=trained_model_h5, 
        # train_data=train_data, 
        test_data=test_data,
        tc1_default_model=tc1_default_model)

    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass

