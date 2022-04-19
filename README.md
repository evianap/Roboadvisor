## Roboadvisor
Roboadvisor is a stock advisory service that takes inputs from the user in order to determine user’s preferred diversification strategy and Investment goals

## About the Project
In this project, we capture the user’s inputs through StreamLit to determine their goals and risk appetite.  We do so by allowing the user to pick and choose between stocks/cryptocurrencies between the year 2021-2022 (present day).  Then, the machine learning robot will run it's code and use ML models to output a table of data based on the csv file of top 50 stock/cyrptocurrency data.  Lastly, the buyer will then be asked if they want to buy/not buy the stock/cryptocurrency.  

## Getting Started
# Installation & Technologies 
This project leverages python 3.8.8 with the following packages:

Python 
Pandas 
Jupyter Lab 
Alphavantage API 
Matplotlib
Streamlit

## Installation Guide Python (MacOS)
### Install Anaconda 
 1. Install [Anaconda](https://www.anaconda.com/products/individual) 
 2. Open up GitBash(Windows) or Terminal(Mac)
 3. Type ```conda update conda```to update Conda
 4. Type ```conda update anaconda```to Update Anaconda
 5. Type ```conda -n dev python=3.9 anaconda```
 6. Type ```conda activate dev```to activate conda
 7. Install a dev environement kernel by typing ```python -m ipykernel install --user --name dev```
 8. Install a mode environement by typing ```conda install -c conda-forge nodejs```
 9. Launch JupyterLab by typing ```jupyter lab```
 
 
 
 ### Install the Request and Json Library 
 We will use the following Python modules and libraries to facilitate API requests:

 1. OS: The OS module comes under Python's standard utility models and provides functions for interacting with the computer's operating system. The OS module does          not require a separate download.
 2. Requests: The Python Requests library helps you access data via APIs.
 3. JSON: This library puts the response (that is, the data) from an API into a human-readable format.

 To install the Requests library, check that your development environment is active, and then run the following command:
  
  ```conda install -c anaconda requests```

 To install the JSON library, check that your development environment is active, and then run the following command:
   
   ```conda install -c jmcmurray json```

## Import the required libraries and dependencies
```
import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import numpy as np

%matplotlib inline
```
We need also to import the relevant alpha vantage libraries :
```
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
````
Finally, we need to import required modules for Linear Regression, Metrics and Classifier: 
````
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

