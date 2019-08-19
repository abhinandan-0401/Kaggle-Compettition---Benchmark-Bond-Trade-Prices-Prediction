from flask import Flask,request,jsonify,abort
import logging,os,sys,json,requests
#from data_preds import get_data1_pred, get_data2_pred, get_data3_pred
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime,sys,os
import pickle
from sklearn.externals import joblib

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

# For Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from itertools import combinations

import warnings
warnings.filterwarnings("ignore")

from numpy import array
from keras.models import Model,Sequential,load_model,model_from_json
from keras.layers import LSTM, Input, Flatten, Dense, Dropout, Bidirectional
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.layers.merge import concatenate
from keras.utils import plot_model

import tensorflow as tf

app = Flask(__name__)

logging.info("Starting Prediction Services")

ROOT_DIR = '.'
DEBUG_MODE='true'

log_filename=os.path.join(ROOT_DIR,"pred_service.log")
file_logger = logging.FileHandler(filename=log_filename)
stdout_logger = logging.StreamHandler(sys.stdout)
loggers = [file_logger,stdout_logger]

logging.basicConfig(level=logging.DEBUG, 
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s: %(message)s', 
                    handlers=loggers)

def load_models():
    logging.info("Entering get_model function")
    models={}
    for i in range(1,4):
        file_name = 'model' + str(i)
        logging.info(file_name)
        try:
            with open('Models/'+file_name+'_json.json', 'r') as f:
                model = model_from_json(f.read())
        except Exception as e:
            logging.info("Error in load model json: "+str(e))
        
        logging.info("Loaded model json from disk")
        try:
            model.load_weights('Models/'+file_name+'_weights.h5')
        except Exception as e:
            logging.info("Error in load model weights: "+str(e))
        models[str(i)] = model
    logging.info("Exiting get_model function")
    return models

models = {}
models = load_models()
graph = tf.get_default_graph()

def split_sequence(sequence, n_steps):
    logging.info("Entering split_sequence function")
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    logging.info("Exiting split_sequence function")
    return array(X), array(y)

def data_reshaper(X,label,type_data):
    logging.info(label)
    if label != "None":
        scaler_filename = "Scalers/scaler3_"+label+".save"
        try:
            scaler = joblib.load(scaler_filename)
        except Exception as e:
            logging.info("Error in reading scaler: ",str(e))
    
    try:
        X_test = scaler.transform(X)
    except Exception as e:
        logging.info("Error in scaler transform data3: ",str(e))
    
    if (type_data == "timeseries"):
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    return X_test

def prep_data1(test):
    logging.info("Entering prep_data1 function")
    test_df = pd.DataFrame(test)
    
    scaler_filename = "Scalers/scaler1.save"
    try:
        scaler = joblib.load(scaler_filename)
    except Exception as e:
        logging.info("Error in reading scaler: ",str(e))
    
    try:
        X_test = test_df["data"].values
        X_test = scaler.transform(X_test.reshape((X_test.shape[0],1)))
    except Exception as e:
        logging.info("Error in scaler transform data1: ",str(e))
    
    X_test, y_test = split_sequence(X_test, 75)
    
    try:
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    except Exception as e:
        logging.info("Error in reshaping data1: ",str(e))
    
    logging.info("Exiting prep_data1 function")
    return X_test, y_test

def prep_data2(test):
    logging.info("Entering prep_data2 function")
    test_df = pd.DataFrame(test)
    
    scaler_filename = "Scalers/scaler2.save"
    try:
        scaler = joblib.load(scaler_filename)
    except Exception as e:
        logging.info("Error in reading scaler: ",str(e))
    
    try:
        X_test = test_df["data"].values
        X_test = scaler.transform(X_test.reshape((X_test.shape[0],1)))
    except Exception as e:
        logging.info("Error in scaler transform data2: ",str(e))
    
    X_test, y_test = split_sequence(X_test, 50)
    
    try:
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    except Exception as e:
        logging.info("Error in reshaping data2: ",str(e))
    logging.info("Exiting prep_data2 function")
    return X_test, y_test

def prep_data3(test):
    logging.info("Entering prep_data3 function")
    test = pd.DataFrame(test)
    time_cols = ['received_time_diff_last1',
       'trade_price_last1', 'trade_size_last1', 'trade_type_last1',
       'curve_based_price_last1', 'received_time_diff_last2',
       'trade_price_last2', 'trade_size_last2', 'trade_type_last2',
       'curve_based_price_last2', 'received_time_diff_last3',
       'trade_price_last3', 'trade_size_last3', 'trade_type_last3',
       'curve_based_price_last3', 'received_time_diff_last4',
       'trade_price_last4', 'trade_size_last4', 'trade_type_last4',
       'curve_based_price_last4', 'received_time_diff_last5',
       'trade_price_last5', 'trade_size_last5', 'trade_type_last5',
       'curve_based_price_last5', 'received_time_diff_last6',
       'trade_price_last6', 'trade_size_last6', 'trade_type_last6',
       'curve_based_price_last6', 'received_time_diff_last7',
       'trade_price_last7', 'trade_size_last7', 'trade_type_last7',
       'curve_based_price_last7', 'received_time_diff_last8',
       'trade_price_last8', 'trade_size_last8', 'trade_type_last8',
       'curve_based_price_last8', 'received_time_diff_last9',
       'trade_price_last9', 'trade_size_last9', 'trade_type_last9',
       'curve_based_price_last9', 'received_time_diff_last10',
       'trade_price_last10', 'trade_size_last10', 'trade_type_last10',
       'curve_based_price_last10']
    num_cols = ['weight','current_coupon','time_to_maturity',
                'reporting_delay','trade_size','curve_based_price']
    cat_cols = ['is_callable','trade_type']
    
    for col in cat_cols:
        logging.info(col, " : ",test[col].nunique())
        
    id_cols = test.columns[test.columns.str.contains(pat = "id")]
    ts_cols = ["trade_size_last","trade_price_last",
               "received_time_diff_last","trade_type_last",
               "curve_based_price_last"]
    
    working_cols = [col for col in test.columns if col not in id_cols]
    working_cols = [col for col in working_cols if col not in test.columns[test.columns.str.contains(pat = "_last")]]
    
    test_main = pd.get_dummies(test[working_cols].drop(columns=["trade_price","weight"], axis=1), columns=cat_cols).values
    logging.info(test_main.shape)
    test_weights = test["weight"].values
    
    ts_data = {}
    for col in ts_cols:
        ts_data[col.split("_last")[0]] = np.fliplr(test[test.columns[test.columns.str.contains(pat = col)]].values)
    
    logging.info("Exiting prep_data3 function")
    return (data_reshaper(ts_data["trade_size"],"trade_size","timeseries"),
            data_reshaper(ts_data["received_time_diff"],"received_time_diff","timeseries"),
            data_reshaper(ts_data["trade_type"],"trade_type","timeseries"),
            data_reshaper(ts_data["curve_based_price"],"curve_based_price","timeseries"),
            data_reshaper(ts_data["trade_price"],"trade_price","timeseries"),
            data_reshaper(test_main,"main","general"))
    

def get_data1_pred(data_dict):
    logging.info("Entering get_data1_pred function")
    X_test, y_test = prep_data1(data_dict["data"])
    
    try:
        y_pred = models["1"].predict(X_test).tolist()
    except Exception as e:
        logging.info("Error in predicting data1: ",str(e))
    
    plt.plot(y_test)
    plt.plot(y_pred)
    
    plt.title('Model plot for data1')
    plt.legend(['Actual', 'Predicted'], loc='upper right');
    
    plt.savefig('Plots/Prediction_Evaluation/Model_Plot_Data1.png')
    plt.close()
    
    results = {"Predictions":y_pred}
    logging.info("Exiting get_data1_pred function")
    return results
    
def get_data2_pred(data_dict):
    logging.info("Entering get_data2_pred function")
    X_test, y_test = prep_data2(data_dict["data"])
    
    try:
        y_pred = models["2"].predict(X_test).tolist()
    except Exception as e:
        logging.info("Error in predicting data2: ",str(e))
    
    plt.plot(y_test)
    plt.plot(y_pred)
    
    plt.title('Model plot for data2')
    plt.legend(['Actual', 'Predicted'], loc='upper right');
    
    plt.savefig('Plots/Prediction_Evaluation/Model_Plot_Data2.png')
    plt.close()
    
    results = {"Predictions":y_pred}
    logging.info("Exiting get_data2_pred function")
    return results
    
def get_data3_pred(data_dict):
    logging.info("Entering get_data3_pred function")
    X_test = prep_data3(data_dict["data"])
    
    if "trade_price" in data_dict["data"]:
        df = pd.DataFrame(data_dict["data"])
        y_test = df["trade_price"].values
    
    try:
        y_pred = models["3"].predict([X_test[0],X_test[1],
                                     X_test[2],X_test[3],
                                     X_test[4],X_test[5]]).tolist()
    except Exception as e:
        logging.info("Error in predicting data3: ",str(e))
        
    try:
        if "trade_price" in data_dict["data"]:
            plt.plot(y_test)
            plt.plot(y_pred)
        
            plt.title('Model plot for data3')
            plt.legend(['Actual', 'Predicted'], loc='upper right');
        
            plt.savefig('Plots/Prediction_Evaluation/Model_Plot_Data3.png')
            plt.close()
    except Exception as e:
        logging.info("Error in predicting ploting: ",str(e))
    
    try:
        results = {"Predictions":y_pred}
    except Exception as e:
        logging.info("Error in compiling pred_results: ",str(e))
    logging.info("Exiting get_data3_pred function")
    return results

@app.route("/predict_data1", methods=['GET','POST'])
def predict_data1():
    logging.info("Entering predict_data1")
    results1 = {}
    if request.method == 'POST':
        global graph
        with graph.as_default():
            try:
                data_dict = request.get_json(force=True)
                results1 = get_data1_pred(data_dict)
            except Exception as e:
                logging.info("Error in input data1: "+str(e))
    logging.info("Exiting predict_data1")
    return jsonify(results1)
    
@app.route("/predict_data2", methods=['GET','POST'])
def predict_data2():
    logging.info("Entering predict_data2")
    results2 = {}
    if request.method == 'POST':
        global graph
        with graph.as_default():
            try:
                data_dict = request.get_json(force=True)
                results2 = get_data2_pred(data_dict)
            except Exception as e:
                logging.info("Error in input data2: "+str(e))
    logging.info("Exiting predict_data2")
    return jsonify(results2)
    
@app.route("/predict_data3", methods=['GET','POST'])
def predict_data3():
    logging.info("Entering predict_data3")
    results3 = {}
    if request.method == 'POST':
        global graph
        with graph.as_default():
            try:
                data_dict = request.get_json(force=True)
                results3 = get_data3_pred(data_dict)
            except Exception as e:
                logging.info("Error in input data3: "+str(e))
    logging.info("Exiting predict_data3")
    return jsonify(results3)

if __name__=="__main__":
    app.run()