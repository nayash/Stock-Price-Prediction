#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""
Sample python script to show how to use hyperopt for hyperparameter tuning
"""

try:
    from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
except:
    !pip install hyperopt
    from hyperopt import Trials, STATUS_OK, tpe
import numpy as np
import pandas as pd
import os
import sys
import time
import pandas as pd
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
import keras
from keras.callbacks import Callback
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import CSVLogger
from sklearn.model_selection import GridSearchCV
# import psutil
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import logging
import itertools as it

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from keras import backend as K

print("checking if GPU available", K.tensorflow_backend._get_available_gpus())

print("current path", os.getcwd())
INPUT_PATH = os.path.join(PATH_TO_DRIVE_ML_DATA, "inputs")
OUTPUT_PATH = os.path.join(PATH_TO_DRIVE_ML_DATA, "outputs")
LOG_PATH = OUTPUT_PATH
LOG_FILE_NAME_PREFIX = "stock_pred_lstm_"
LOG_FILE_NAME_SUFFIX = ".log"

TIME_STEPS = 90  # 3 months
BATCH_SIZE = 20
stime = time.time()

def print_time(text, stime):
    seconds = (time.time() - stime)
    print(text + " " + str(seconds // 60) + " minutes : " + str(np.round(seconds % 60)) + " seconds")

def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")

def init_logging():
    logging.basicConfig(level=logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(os.path.join(LOG_PATH, LOG_FILE_NAME_PREFIX + get_readable_ctime()+".log")) # "{0}/{1}.log".format(LOG_PATH, LOG_FILE_NAME_PREFIX + get_readable_ctime()))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0] % batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


def build_timeseries(mat, y_col_index, time_steps):
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - time_steps
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((x.shape[0],))

    for i in tqdm(range(dim_0)):
        x[i] = mat[i:time_steps + i]
        y[i] = mat[time_steps + i, y_col_index]
    print("length of time-series i/o {} {}".format(x.shape, y.shape))
    return x, y

stime = time.time()
init_logging()
print(str(os.listdir(INPUT_PATH)))  # ge.us.txt
df_ge = pd.read_csv(os.path.join(INPUT_PATH,"ge.us.txt"), engine='python')
print(str(df_ge.shape))
print(str(df_ge.columns))

train_cols = ["Open", "High", "Low", "Close", "Volume"]
df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
print("Train--Test size {} {}".format(len(df_train), len(df_test)))

# scale the feature MinMax, build array
mat = df_ge.loc[:, train_cols].values

print("Deleting unused dataframes of total size(KB) {}"
      .format((sys.getsizeof(df_ge) + sys.getsizeof(df_train) + sys.getsizeof(df_test)) // 1024))

del df_ge
del df_test
del df_train

csv_logger = CSVLogger(OUTPUT_PATH + 'log_' + get_readable_ctime() + '.log', append=True)

class LogMetrics(Callback):

    def __init__(self, search_params, param, comb_no):
        self.param = param
        self.self_params = search_params
        self.comb_no = comb_no
        
    def on_epoch_end(self, epoch, logs):
        for i, key in enumerate(self.self_params.keys()):
            logs[key] = self.param[key]
        logs["combination_number"] = self.comb_no


def data_dummy():
    return None, None, None, None


def data(batch_size, time_steps):
    global mat

    BATCH_SIZE = batch_size
    TIME_STEPS = time_steps
    x_train, x_test = train_test_split(mat, train_size=0.8, test_size=0.2, shuffle=False)

    # scale the train and test dataset
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)

    x_train_ts, y_train_ts = build_timeseries(x_train, 3, TIME_STEPS)
    x_test_ts, y_test_ts = build_timeseries(x_test, 3, TIME_STEPS)
    x_train_ts = trim_dataset(x_train_ts, BATCH_SIZE)
    y_train_ts = trim_dataset(y_train_ts, BATCH_SIZE)
    x_test_ts = trim_dataset(x_test_ts, BATCH_SIZE)
    y_test_ts = trim_dataset(y_test_ts, BATCH_SIZE)
    print("Test size(trimmed) {}, {}".format(x_test_ts.shape, y_test_ts.shape))

    print("Are any NaNs present in train/test matrices?{0},{1}".format(str(np.isnan(x_train).any()),
                                                                       str(np.isnan(x_test).any())))
    return x_train_ts, y_train_ts, x_test_ts, y_test_ts


search_space = {
    'batch_size': hp.choice('bs', [30,40,50,60,70]),
    'time_steps': hp.choice('ts', [30,50,60,80,90]),
    'lstm1_nodes':hp.choice('units_lsmt1', [70,80,100,130]),
    'lstm1_dropouts':hp.uniform('dos_lstm1',0,1),
    'lstm_layers': hp.choice('num_layers_lstm',[
        {
            'layers':'one', 
        },
        {
            'layers':'two',
            'lstm2_nodes':hp.choice('units_lstm2', [20,30,40,50]),
            'lstm2_dropouts':hp.uniform('dos_lstm2',0,1)  
        }
        ]),
    'dense_layers': hp.choice('num_layers_dense',[
        {
            'layers':'one'
        },
        {
            'layers':'two',
            'dense2_nodes':hp.choice('units_dense', [10,20,30,40])
        }
        ]),
    "lr": hp.uniform('lr',0,1),
    "epochs": hp.choice('epochs', [30, 40, 50, 60, 70]),
    "optimizer": hp.choice('optmz',["sgd", "rms"])
}


def create_model_hypopt(params):
    print("Trying params:",params)
    batch_size = params["batch_size"]
    time_steps = params["time_steps"]
    x_train_ts, y_train_ts, x_test_ts, y_test_ts = data(batch_size, time_steps)
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(params["lstm1_nodes"], batch_input_shape=(batch_size, time_steps, x_train_ts.shape[2]), dropout=params["lstm1_dropouts"],
                        recurrent_dropout=params["lstm1_dropouts"], stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))  
    # ,return_sequences=True #LSTM params => dropout=0.2, recurrent_dropout=0.2
    if params["lstm_layers"]["layers"] == "two":
        lstm_model.add(LSTM(params["lstm_layers"]["lstm2_nodes"], dropout=params["lstm_layers"]["lstm2_dropouts"]))
    else:
        lstm_model.add(Flatten())

    if params["dense_layers"]["layers"] == 'two':
        lstm_model.add(Dense(params["dense_layers"]["dense2_nodes"], activation='relu'))
    
    lstm_model.add(Dense(1, activation='sigmoid'))

    lr = params["lr"]
    epochs = params["epochs"]
    if params["optimizer"] == 'rms':
        optimizer = optimizers.RMSprop(lr=lr)
    else:
        optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)  # binary_crossentropy
    history = lstm_model.fit(x_train_ts, y_train_ts, epochs=epochs, verbose=2, batch_size=batch_size,
                             validation_data=[x_test_ts, y_test_ts],
                             callbacks=[LogMetrics(search_space, params, -1), csv_logger])
    # for key in history.history.keys():
    #     print(key, "--",history.history[key])
    # get the highest validation accuracy of the training epochs
    val_error = np.amin(history.history['val_loss']) 
    print('Best validation error of epoch:', val_error)
    return {'loss': val_error, 'status': STATUS_OK, 'model': lstm_model} # if accuracy use '-' sign
    

trials = Trials()
best = fmin(create_model_hypopt,
    space=search_space,
    algo=tpe.suggest,
    max_evals=2000,
    trials=trials)

pickle.dump(best, open(os.path.join(OUTPUT_PATH,"hyperopt_res"),"wb"))

print(best)
print_time("program completed in", stime)
