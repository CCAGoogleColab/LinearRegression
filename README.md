# Preview

In this lab, you’ll use a linear regression machine learning algorithm to estimate a person’s medical insurance cost with his or her BMI.(Body Mass Index)

# Getting set up

	Import Libraries for Linear Regression
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.linear_model import LinearRegression
	from sklearn.model_selection import train_test_split 

* Pandas is a fast, powerful, flexible and easy to use open source data analysis tool, built within the Python programming language.
* Numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
* Matplotlib is a plotting library for the Python programming language.
* Import these libraries as pd, np, and plt for simplification and efficiency.
* Sklearn (Scikit-Learn) is a machine learning library for the Python programming language that features various classification, regression and clustering algorithms.
* LinearRegression and train_test_split is used for machine learning and splitting the raw data for training.
	Import Data File from Google Drive
	(Code to read csv file into Colaboratory:)
	!pip install -U -q PyDrive
	from pydrive.auth import GoogleAuth
	from pydrive.drive import GoogleDrive
	from google.colab import auth
	from oauth2client.client import GoogleCredentials'

	(Authenticate and create the PyDrive client.)
	auth.authenticate_user()
	gauth = GoogleAuth()
	gauth.credentials = GoogleCredentials.get_application_default()
	drive = GoogleDrive(gauth)
