import numpy as np
import sklearn.impute
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline


X, y = sklearn.datasets.load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, shuffle=True, random_state=16, test_size=0.15, stratify=y
)

na_inds = np.random.random(X_train.shape) < 0.3
X_train_corrupted = np.where(na_inds, np.nan, X_train)



# 1) Simple imputation (considers only a single feature at time)
# Note: for categorical data, use strategy="most_frequent"
pipeline = sklearn.pipeline.Pipeline([
    ("imputer", sklearn.impute.SimpleImputer(missing_values=np.nan, strategy="mean")),
    ("log_reg", sklearn.linear_model.LogisticRegression(multi_class="multinomial")),
])

pipeline.fit(X_train, y_train)
y_preds = pipeline.predict(X_test)
acc = sklearn.metrics.accuracy_score(y_test, y_preds)
print(f"Accuracy: {acc:.3f}")


pipeline.fit(X_train_corrupted, y_train)
y_preds = pipeline.predict(X_test)
acc = sklearn.metrics.accuracy_score(y_test, y_preds)
print(f"Accuracy: {acc:.3f}")


# 2) KNN imputation
pipeline = sklearn.pipeline.Pipeline([
    ("imputer", sklearn.impute.KNNImputer(n_neighbors=4, weights="distance")),
    ("log_reg", sklearn.linear_model.LogisticRegression(multi_class="multinomial")),
])

pipeline.fit(X_train, y_train)
y_preds = pipeline.predict(X_test)
acc = sklearn.metrics.accuracy_score(y_test, y_preds)
print(f"Accuracy: {acc:.3f}")


pipeline.fit(X_train_corrupted, y_train)
y_preds = pipeline.predict(X_test)
acc = sklearn.metrics.accuracy_score(y_test, y_preds)
print(f"Accuracy: {acc:.3f}")
