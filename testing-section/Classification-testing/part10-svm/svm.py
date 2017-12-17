# Get the dataset and split into dependent array and independent matrix
import pandas as pd
dataset = pd.read_csv('/home/felipe/Documentos/Machine Learning A-Z/' + 
	'Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')
ind_var = dataset.iloc[:, :-1].values
dep_var = dataset.iloc[:,  -1].values

# Search for missing values
# No missing data.

# Check for categorical/hierarchical parameters
# Column #0 is useless and #1 do not contribute too much.
ind_var = ind_var[:, 2:(ind_var.shape[1] + 1)]

# Feature scaling?
# Side note: if feature scaling is not applied, the fitting process will
# be extremely slow.
from sklearn.preprocessing import StandardScaler as skl_ss
ind_var = skl_ss().fit_transform(ind_var.astype(float))

# Split the dataset into test and train sets
from sklearn.model_selection import train_test_split as skl_tts
ind_train_set, ind_test_set, dep_train_set, dep_test_set = skl_tts(ind_var, dep_var, test_size = 0.25)

# Fit the model
from sklearn.svm import SVC as skl_svc
lin_classifier = skl_svc(kernel = 'linear') # Linear kernel SVM is almost the same as logistic regression
rbf_classifier = skl_svc(kernel = 'rbf', gamma = 'auto') # RBF = Radial Basis Function
sig_classifier = skl_svc(kernel = 'sigmoid', gamma = 'auto', coef0 = 0.0)
pol_classifier = skl_svc(kernel = 'poly', degree = 3, gamma = 'auto', coef0 = 0.0)

lin_classifier.fit(ind_train_set, dep_train_set) 
rbf_classifier.fit(ind_train_set, dep_train_set) 
sig_classifier.fit(ind_train_set, dep_train_set) 
pol_classifier.fit(ind_train_set, dep_train_set) 

# Predict the test set
lin_prediction = lin_classifier.predict(ind_test_set)
rbf_prediction = rbf_classifier.predict(ind_test_set)
sig_prediction = sig_classifier.predict(ind_test_set)
pol_prediction = pol_classifier.predict(ind_test_set)

# Create confusion matrix to verify the results
from sklearn.metrics import confusion_matrix as skl_cm
print 'Linear Kernel confusion matrix'
skl_cm(y_true = dep_test_set, y_pred = lin_prediction)
print 'RBF (Gaussian) Kernel confusion matrix'
skl_cm(y_true = dep_test_set, y_pred = rbf_prediction)
print 'Sigmoid Kernel confusion matrix'
skl_cm(y_true = dep_test_set, y_pred = sig_prediction)
print 'Polynomial (3degree) Kernel confusion matrix'
skl_cm(y_true = dep_test_set, y_pred = pol_prediction)

# =============================================
# Disclaimer: All the plotting code below is not mine.
# Font: Machine Learning A-Z™: Hands-On Python & R In Data Science - kernel_svm.py source code 
# =============================================
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt 
import numpy as np

X_set, y_set = ind_train_set, dep_train_set
classifier = pol_classifier

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
# ============================================