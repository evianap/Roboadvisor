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
```
`````




## Usage


## Dashboard Demonstration

## Contributers 

Harini Atmakuri
Amanda Hum
Jose (last name)
Amine Baite
Alexis Rose Garcia Alexisg324@gmail.com https://www.linkedin.com/in/alexis-rose-garcia
Kyle Huber kyhuber@gmail.com https://www.linkedin.com/in/huberkyle/

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

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM Hit Return for more, or q (and Return) to quit: LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

## Workcited:

Financial Technology Bootcamp
UCB-Coding-Bootcamp (2021-2022). Module 1-16. UC Berkeley Fintech Extension. https://courses.bootcampspot.com/
