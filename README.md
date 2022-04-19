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
Command + Space
Enter Terminal in search bar and press enter
Terminal should open
Open Homebrew Installation website in browser (https://brew.sh/)
Copy installation code at the bottom of the page
Once Homebrew is installed, install Python
Open Terminal
Enter "brew install python3" into the CLI to install Python
Installing Anaconda with Python (MacOS):

Enter "conda create -n dev python=3.7 anaconda" into the command line terminal
Return and type Y when prompted
Open environment by entering "conda activate dev"
Enable terminal commands through conda by enter "echo $ {SHELL}" to check BASH/ZSH environment
Depending on if BASH/ZSH, type "conda init bash or ZSH" to activate conda terminal commands
Close environment by entering "conda deactivate"

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

