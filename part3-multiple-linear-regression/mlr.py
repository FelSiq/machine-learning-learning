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

# Not sure about feature scalling.
# Split the dataset into the train and test set
from sklearn.model_selection import train_test_slip as sklpp_tts
dep_param_train, dep_param_test, ind_param_train, ind_param_test = sklpp_tts(dep_param, ind_param, test_size = 0.2)

# Now the fun starts.
# Build the multivariable linear regression.
