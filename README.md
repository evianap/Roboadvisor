## Roboadvisor
Roboadvisor is a stock advisory service that takes inputs from the user in order to determine the user’s preferred diversification strategy and Investment goals

## About the Project
The goal of the project is to capture the user’s inputs through StreamLit to determine their goals and risk appetite, and provide recommendation on the stocks/cryptocurrencies that they can invest in.  We do so by allowing the user to pick and choose between stocks/cryptocurrencies. Then, the machine learning robot will run it's code and use ML models to output a table of data detailing which stocks/cryptocurrencies in the portfolio will make money. Lastly, the buyer will then be asked if they want to buy/not buy the stock/cryptocurrency and orders will be placed.
Minimum Viable Product(MVP): The Robotrader is a service that uses multiple Machine Learning models to recommend whether or not to buy a stock based on the following: Historical prices, Macroeconomic factors, and Company fundamentals. To acheive the MVp project objectives, we developed a model that will allow users to select a stock and provide their expectation of the macro economic and company fundamentals. Then, the machine learning robot will run it's code and use ML models to output a recomemndation detailing the closing price of the stock in question. 

## Getting Started
# Installation & Technologies 
This project leverages python 3.8.8 with the following packages:

Python 
Pandas 
Jupyter Lab 
Alphavantage API 
Matplotlib
Streamlit

**Approach:** 

We followed the below steps in meeting the project objectives for the MVP

1. Data Collection: Build APIs to gather data on specific stocks/cryptos and macro economic indicators.
      Source: Alpha Vantage
      Pulled historical data on the following indicators:
      EPS
      Stock Price (on top 50 stocks in the S&P 500)
      Cryptocurrency Price
      Inflation
      Consumer Sentiment
2. Data Analysis: Analyze the data trends and the correlation between the indicators. Below is an analysis of the correlation between different features that make up the ctock price, and ana anlsysi of the closing price of the stocks. 
   <img width="473" alt="Screen Shot 2022-04-24 at 8 57 37 AM" src="https://user-images.githubusercontent.com/96159292/164985383-b769de72-57b1-40d5-b8e0-cdf70887e8ac.png">
<img width="398" alt="Screen Shot 2022-04-24 at 8 58 42 AM" src="https://user-images.githubusercontent.com/96159292/164985390-3d2c3864-8077-4b2f-a86a-3ed15b59c4f3.png">
4. Structure & Format Data: Work through data to ensure that periods match up and that it is accepted into the ML models.
<img width="671" alt="Screen Shot 2022-04-24 at 9 08 13 AM" src="https://user-images.githubusercontent.com/96159292/164985529-b09aaac0-f41c-45c4-b371-f21184e6161f.png">
6. ML: Run the data through multiple ML models, train the models, compare model performance. More specifically, we looked at the problem from a classification problem perspective as well as a Regression problem perspective, and selected the best ML model for each of the problems. <img width="882" alt="Screen Shot 2022-04-24 at 9 18 23 AM" src="https://user-images.githubusercontent.com/96159292/164986039-194d93aa-7023-4798-b0a0-4b9fadd15254.png">
 
<img width="877" alt="Screen Shot 2022-04-24 at 9 18 07 AM" src="https://user-images.githubusercontent.com/96159292/164986037-cafd22e4-0e74-4d72-a9b2-8189a2674053.png">
8. User Input and Ouput: Capture user input through Streamlit to understand Investors expectation of the various macroeconomic and company fundametals to provide the expected price for stock to enable user to make the decision.

   a.  Robotrader will recieve user input through streamllit 
   
<img width="491" alt="Screen Shot 2022-04-24 at 7 58 52 AM" src="https://user-images.githubusercontent.com/96159292/164986140-b9c2de42-336d-4750-84b8-d337e2414ce0.png">

   b. Robot will determine the price of the stock:
   
 <img width="488" alt="Screen Shot 2022-04-24 at 9 40 27 AM" src="https://user-images.githubusercontent.com/96159292/164986963-9628f784-a6b1-48fd-8890-80036483522d.png">

**Next steps:** 

To further develop the product to realize the complete project objectives are provided below

<img width="749" alt="Screen Shot 2022-04-24 at 9 25 23 AM" src="https://user-images.githubusercontent.com/96159292/164986293-ece3a75d-60b6-4d15-88fc-a3140da0996e.png">

**Further Details:** 
  **Presentation** https://docs.google.com/presentation/d/1rYWpFhiXFshuFSV1FPA2J8I28cMPgwta2fUc7Wr2Jbc/edit#slide=id.g125114ac3ca_0_0

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
import seaborn as sns
%matplotlib inline
```
We need also to import the relevant alpha vantage libraries :
```
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
````
We need to import required modules for Classificatgion, Regression, Neural Networks, and Metrics: 
````
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
```
````
## Contributors 

Ram Atmakuri 
ram.atmakuri@outlook.com 
https://www.linkedin.com/in/ramatmakuri99

Amanda Hum
amanda.m.hum@gmail.com
https://www.linkedin.com/in/amandamhum

Enrique Viana
EMAIL
https://www.linkedin.com/in/enriqueviana

Mohammed Amine Baite
aminebaite@gmail.com
https://www.linkedin.com/in/amine-baite-3972a71b8/

Alexis Rose Garcia 
Alexisg324@gmail.com 
https://www.linkedin.com/in/alexis-rose-garcia

Kyle Huber 
kyhuber@gmail.com 
https://www.linkedin.com/in/huberkyle/

## License
license() A. HISTORY OF THE SOFTWARE ==========================

Python was created in the early 1990s by Guido van Rossum at Stichting Mathematisch Centrum (CWI, see http://www.cwi.nl) in the Netherlands as a successor of a language called ABC. Guido remains Python's principal author, although it includes many contributions from others.

In 1995, Guido continued his work on Python at the Corporation for National Research Initiatives (CNRI, see http://www.cnri.reston.va.us) in Reston, Virginia where he released several versions of the software.

In May 2000, Guido and the Python core development team moved to BeOpen.com to form the BeOpen PythonLabs team. In October of the same year, the PythonLabs team moved to Digital Creations, which became Zope Corporation. In 2001, the Python Software Foundation (PSF, see https://www.python.org/psf/) was formed, a non-profit organization created specifically to own Python-related Intellectual Property. Zope Corporation was a sponsoring member of the PSF.

All Python releases are Open Source (see http://www.opensource.org for the Open Source Definition). Historically, most, but not all, Python Hit Return for more, or q (and Return) to quit: releases have also been GPL-compatible; the table below summarizes the various releases.

Release         Derived     Year        Owner       GPL-
                from                                compatible? (1)

0.9.0 thru 1.2              1991-1995   CWI         yes
1.3 thru 1.5.2  1.2         1995-1999   CNRI        yes
1.6             1.5.2       2000        CNRI        no
2.0             1.6         2000        BeOpen.com  no
1.6.1           1.6         2001        CNRI        yes (2)
2.1             2.0+1.6.1   2001        PSF         no
2.0.1           2.0+1.6.1   2001        PSF         yes
2.1.1           2.1+2.0.1   2001        PSF         yes
2.1.2           2.1.1       2002        PSF         yes
2.1.3           2.1.2       2002        PSF         yes
2.2 and above   2.1.1       2001-now    PSF         yes
Footnotes:

(1) GPL-compatible doesn't mean that we're distributing Python under the GPL. All Python licenses, unlike the GPL, let you distribute a modified version without making your changes open source. The Hit Return for more, or q (and Return) to quit: GPL-compatible licenses make it possible to combine Python with other software that is released under the GPL; the others don't.

(2) According to Richard Stallman, 1.6.1 is not GPL-compatible, because its license has a choice of law clause. According to CNRI, however, Stallman's lawyer has told CNRI's lawyer that 1.6.1 is "not incompatible" with the GPL.

Thanks to the many outside volunteers who have worked under Guido's direction to make these releases possible.

B. TERMS AND CONDITIONS FOR ACCESSING OR OTHERWISE USING PYTHON
Starting with Python 3.8.6, examples, recipes, and other code in the documentation are dual licensed under the PSF License Version 2 and the Zero-Clause BSD license.

Some software incorporated into Python is under different licenses. Hit Return for more, or q (and Return) to quit: The licenses are listed with code falling under that license.

This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and the Individual or Organization ("Licensee") accessing and otherwise using Python 3.10.2 software in source or binary form and its associated documentation.

Subject to the terms and conditions of this License Agreement, PSF hereby grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce, analyze, test, perform and/or display publicly, prepare derivative works, distribute, and otherwise use Python 3.10.2 alone or in any derivative version, provided, however, that PSF's License Agreement and PSF's notice of copyright, i.e., "Copyright © 2001-2022 Python Software Foundation; All Rights Reserved" are retained in Python 3.10.2 alone or in any derivative version prepared by Licensee.

In the event Licensee prepares a derivative work that is based on or incorporates Python 3.10.2 or any part thereof, and wants to make the derivative work available to others as provided herein, then Licensee hereby agrees to include in any such work a brief summary of the changes made to Python 3.10.2.

PSF is making Python 3.10.2 available to Licensee on an "AS IS" basis. PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON 3.10.2 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON 3.10.2 FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 3.10.2, OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

This License Agreement will automatically terminate upon a material breach of its terms and conditions.

Nothing in this License Agreement shall be deemed to create any relationship of agency, partnership, or joint venture between PSF and Licensee. This License Agreement does not grant permission to use PSF trademarks or trade name in a trademark sense to endorse or promote products or services of Licensee, or any third party.

By copying, installing or otherwise using Python 3.10.2, Licensee agrees to be bound by the terms and conditions of this License Agreement.

ZERO-CLAUSE BSD LICENSE FOR CODE IN THE PYTHON DOCUMENTATION
Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

## Workcited:

Financial Technology Bootcamp
UCB-Coding-Bootcamp (2021-2022). Module 1-16. UC Berkeley Fintech Extension. https://courses.bootcampspot.com/

Alpha Vantage (API): 
Alpha Vantage (2022). www.alphavantage.co
