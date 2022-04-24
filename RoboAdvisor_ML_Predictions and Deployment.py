#!/usr/bin/env python
# coding: utf-8

# # Link to the earlier steps: In the earlier steps, we prepared the data and analyzed differnet calssification and regression models. As the optimal models are selected in the earlier steps, we will perform the steps as below here:
# 1. Based on the final files exported, we will reperform the finalist classification and regression models
# 2. Store the final models information using Pickle
# 3. Load the stored model and test it that it gives the same result as the originally trained model
# 4. Perform Prediction based on data in a csv file
# Please refer to the "Data Preparation, Analysis and Artificial Intelligence or Machine Learning Analysis" file for the earlier steps referred to above.

# In[18]:


# Import the required libraries and dependencies
import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import numpy as np
import time
#get_ipython().run_line_magic('matplotlib', 'inline')

# In[19]:


#Import tha required libraries for datapreparation, analysis and preprocessing
import sklearn
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# In[20]:

#Import Required modules for Linear Regression
from sklearn.model_selection import train_test_split

# Import Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Import different Models -  1. linera_model - LogisticRegression
#Import different Models -  5. svm - LinearSVC

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC 
#from sklearn.preprocessing import GetDummies
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#Linear Regression Library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# In[21]:


#Import tha relevant alpha vantage libraries
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import streamlit as st

# In[22]:
#set key
key = '32UEEELX55R5WBXX'
start_date = '2015-04-01' 


# # Classification

# ## Read Data from preprocessed file, Prepare Data and Run Classification ML with the selected AI/ML Model -  Random Forest Classifier

# In[23]:
#Load the stock analysis data for classification analysis
csv_path = Path("Resources/stock_analysis_data.csv")
stock_analysis_data = pd.read_csv(csv_path, index_col = 'date')
print(stock_analysis_data)

csv_path_ticker = Path("Resources/Tickers.csv") 
Tickers= pd.read_csv(csv_path_ticker)
print(Tickers)
# In[24]:
        
# Listing of the features columns 
FEATURES = list(stock_analysis_data.iloc[:,:-1])
print(FEATURES)

# In[25]:


# Prepare test and train data
test_frac = 0.2
X = stock_analysis_data[FEATURES]
y = stock_analysis_data['Target']
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = test_frac)
print(x_train)


# In[26]:
stock_logistic_regression_model = LogisticRegression(solver = 'liblinear')
stock_logistic_regression_model.fit(x_train, y_train)
y_pred = stock_logistic_regression_model.predict(x_test)


# In[27]:



acc = accuracy_score (y_test, y_pred, normalize = True)
num_acc = accuracy_score (y_test, y_pred, normalize = False)
prec = precision_score (y_test, y_pred)
recall = recall_score (y_test, y_pred)

print(acc)
print(prec)
print(recall)


# In[30]:


# Define the RandomForestClassifier Model, Fit and Predict
stock_random_forest_classifirer_model = RandomForestClassifier()
stock_random_forest_classifirer_model.fit(x_train, y_train)
y_pred = stock_random_forest_classifirer_model.predict(x_test)


# In[31]:


# Analyze the resuls of the RandomForestClassifier Model
rf_acc = accuracy_score (y_test, y_pred, normalize = True)
rf_num_acc = accuracy_score (y_test, y_pred, normalize = False)
rf_prec = precision_score (y_test, y_pred)
rf_recall = recall_score (y_test, y_pred)

print(rf_acc)
print(rf_prec)
print(rf_recall)


# ## Store the RamdomClassifier model and Reload the saved model

# In[32]:


# Import Pickle 
import pickle


# In[34]:


# Save the model
filename = 'random_forest_classifier.sav'
pickle.dump(stock_random_forest_classifirer_model, open(filename, 'wb'))


# In[38]:


#Load the saved model
loaded_model = pickle.load(open('random_forest_classifier.sav','rb'))


# In[49]:


#Predict using the loaded model and compare with the orginal model
y1_pred = loaded_model.predict(x_test)


# In[50]:


print(y_test)


# In[40]:


#Predict using the loaded model and compare with the orginal model to confirm that the original model is the same as the loaded model.
rf_acc = accuracy_score (y_test, y_pred, normalize = True)
rf_num_acc = accuracy_score (y_test, y_pred, normalize = False)
rf_prec = precision_score (y_test, y_pred)
rf_recall = recall_score (y_test, y_pred)

print(rf_acc)
print(rf_prec)
print(rf_recall)


# ## Predict based on CSV file using the Loaded Model

# In[42]:


# Load the file (Supposedly obtained from the Investor) from csv format
csv_path = Path("Resources/Classifier_Test.csv")
stock_test = pd.read_csv(csv_path, index_col = 'date', header=1)


# In[43]:


#Review the analysis file
print(stock_test)


# In[47]:


# Predict based on loaded model
prediction = loaded_model.predict(stock_test)


# In[48]:


# Print
print(prediction)


# In[53]:


#Result of the analysis
if prediction [0] == 0:
    print('Do not buy the stocks')
else:
    print('Buy the stocks')


# # Regression

# ## Read Data from preprocessed file, Prepare Data and Run Regression ML with the selected AI/ML Model -  Random Forest Regression

# In[85]:


#Load the stock analysis data for classification analysis
csv_path = Path("Resources/stock_analysis_data_lr.csv")
stock_analysis_data_lr = pd.read_csv(csv_path, index_col = 'date')
print(stock_analysis_data)


# In[86]:
# Listing of the features columns 
FEATURES = list(stock_analysis_data_lr.iloc[:,:-1])
print(FEATURES)
st.title('Welcome to RoboAdvisor!!!')
st.subheader('Please select a stock from the dropdown below for obtaining estimated prices')
selected_stocks = st.multiselect ("Tickers?", Tickers)
st.write('You selected:')
stock_head = ()
for i in selected_stocks:
    st.write(i)
    stock_head = (f"%-close_{i}-close") 
stock_test_regr_1 = pd.DataFrame(columns = FEATURES)
if len(selected_stocks)!=1:
    st.error('Invalid selection count. Please select upto 1 stock')
else:
    stock_head
    count=0
    EPS = st.number_input(f"please provide your expectation of EPS for {selected_stocks}", key=count)
    stock_test_regr_1.at[0, 'EPS'] = EPS
    count += 1
    treasury_yield = st.number_input(f"please provide your expectation of rate of change in treasury Yield for the period", key=count)
    stock_test_regr_1.at[0, 'treausry_yield'] = treasury_yield
    count += 1
    inflation_expectation = st.number_input(f"please provide your expectation of rate of change in inflation for the period", key=count)
    stock_test_regr_1.at[0, 'inflation_expectation'] = inflation_expectation
    count += 1
    sp500_return = st.number_input(f"please provide your expectation of rate of change in SP500 for the period", key=count)
    stock_test_regr_1.at[0, 'sp500_return'] = sp500_return
    count += 1
    unemployment_rate = st.number_input(f"please provide your expectation of rate of change in unemployment rate for the period", key=count)
    stock_test_regr_1.at[0, 'unemployment_rate'] = unemployment_rate
    count += 1
    consumer_sentiment = st.number_input(f"please provide your expectation of rate of change in consumer sentiment for the period", key=count)
    stock_test_regr_1.at[0, 'consumer_sentiment'] = consumer_sentiment
    stock_test_regr_1.at[0, stock_head] = 1
stock_test_regr_1.fillna(0, inplace = True)
stock_test_regr_1


# In[104]:


# Prepare test and train data
test_frac = 0.2
Xlr = stock_analysis_data_lr[FEATURES]
ylr = stock_analysis_data_lr['value']
x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(Xlr,ylr, test_size = test_frac)
print(x_train_lr)


# In[105]:


# Define the RandomForestRegression Model, Fit and Predict
stock_random_forest_regression_model = RandomForestRegressor()
stock_random_forest_regression_model.fit(x_train_lr, y_train_lr)
ylr_pred = stock_random_forest_regression_model.predict(x_test_lr)
print(ylr_pred)


# In[108]:


r21_score = r2_score(y_test_lr, ylr_pred, multioutput='variance_weighted')
print(r21_score)


# ## Store the RamdomRegression model and Reload the saved model

# In[109]:


#Save the regressor model
filename = 'random_forest_regressor.sav'
pickle.dump(stock_random_forest_regression_model, open(filename, 'wb'))


# In[110]:


#loading the saved model
loaded_model_lr = pickle.load(open('random_forest_regressor.sav','rb'))


# value
# 59.6512
# 136.2333
# 102.7
# 68.166
# 120.1474

# ## Predict based on CSV file using the Loaded Model

# In[111]:


# Load the file (Supposedly obtained from the Investor) from csv format
csv_path = Path("Resources/stock_test_regression.csv")
stock_test_regr = pd.read_csv(csv_path, index_col = 'date', header=1)
print(stock_test_regr)


# In[112]:

#stock_test_regr_1.drop(columns='Target', inplace = True)
# Predict the stock values for the data obtained from the investor
regr_predictions = loaded_model_lr.predict(stock_test_regr_1)
st.write(f"The prices for the {selected_stocks} based on the input parameters provided will be{regr_predictions}")