from flask import Flask,request,jsonify,abort
import logging,os,sys,json,requests

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

print("Starting Prediction Automation Program")

ROOT_DIR = '.'
DEBUG_MODE='true'

input_file = sys.argv[1]
input_dir = "../datasets/"

def get_data():
    filename = input_file + ".csv"
    filepath = input_dir + filename
    
    if int(input_file[-1]) != 3:
        df = pd.read_csv(filepath, header=None, names=["data"])
        train_size = int(df.shape[0] * 0.75)
        test = df.iloc[train_size:,:]
        test = test.to_dict()
    else:
        df = pd.read_csv(filepath)
        train_size = int(df.shape[0] * 0.75)
        test = df.iloc[train_size:,:]
        test = test.to_dict()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    return test

def process_preds():
    headers = {"Content-Type" : "application/json"}
    test_dict = get_data()
    data = {"data" : test_dict}
    
    if int(input_file[-1]) == 1:
        try:
            url = "http://127.0.0.1:5000/predict_data1"
            response = requests.post(url,data=json.dumps(data),headers=headers)
        except Exception as e:
            print("Error in data1 predictions: ",str(e))
    elif int(input_file[-1]) == 2:
        try:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            url = "http://127.0.0.1:5000/predict_data2"
            response = requests.post(url,data=json.dumps(data),headers=headers).json()
        except Exception as e:
            print("Error in data1 predictions: ",str(e))
    else:
        try:
            url = "http://127.0.0.1:5000/predict_data3"
            response = requests.post(url,data=json.dumps(data),headers=headers).json()
        except Exception as e:
            print("Error in data1 predictions: ",str(e))
#    out_df = pd.DataFrame(response)
#    print(out_df.head())
#    out_df.to_csv("Predictions"+input_file+".csv", index=False)
    
    
if __name__=="__main__":
    process_preds()