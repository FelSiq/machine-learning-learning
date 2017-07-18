"""
Simple linear regression: 	y = b0 + b1x1
Multiple Linear Regression:	y = b0 + b1x1 + b2x2 + ... + bnxn
"""

# Section one: import the libraries
import pandas as pd # Data handling

# Section two: Get the dataset
dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
dep_param = dataset.iloc[:, :-1].values
ind_param = dataset.iloc[:,  -1].values

# Section three: No missing values.

# Section four: Found categorical values, let's discretize it
from sklearn.preprocessing import LabelEncoder as sklpp_le, OneHotEncoder as sklpp_ohe
# Explanations:
#	sklpp_le 	-> discretize values
#	sklpp_ohe 	-> create dummy parameters

# Create simple encoder, calling it's construtor method
enc_discretize = sklpp_le()
# Discretize the State column of the dependent parameters subdataset
dep_param[:, 3] = enc_discretize.fit_transform(dep_param[:, 3])
# Now, it's time to create dummy paramaters of the previous decodifycation.
# Call OneHotEncoder constructor method
enc_distodummy = sklpp_ohe(categorical_features = [3])
dep_param = enc_distodummy.fit_transform(dep_param).toarray()

# Remember the Dummy Variable Trap! (Where redundant linear dependency was created on some linear formulae)
# Note: Of course Python MLR take care of this for us, but, anyway, just to
# be sure.
# dep_param = dep_param[:, 1:]

# Not sure about feature scalling.
# Split the dataset into the train and test set
from sklearn.model_selection import train_test_split as sklpp_tts
dep_param_train, dep_param_test, ind_param_train, ind_param_test = sklpp_tts(dep_param, ind_param, test_size = 0.2)

# Now the fun starts.
# Build the multivariable linear regression.
from sklearn.linear_model import LinearRegression as skllm_lr
regressor = skllm_lr()
regressor.fit(dep_param_train, ind_param_train)

# Make predictions
predictions = regressor.predict(dep_param_test)

# No ploting (Because too much dimensions to visualise).

# Is this the optimum model?
# Let's use the backward feature selection method to check it out.
import statsmodels.formula.api as smfa
import numpy as np

# Stop that crazy precision printting
np.set_printoptions(precision = 2)

# Create the linear coefficient column (because statsmodel does not treat this to us)
# The trick is to create a '1' column to the train dataset:
# y = b0*1 + b1x1 + b2x2 + ... + bnxn
dep_param = np.append(
	arr = np.ones(shape = (50, 1)).astype(int), 
	values = dep_param, 
	axis = 1)