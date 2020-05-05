# Classification of Activity based on the EMG Physical Dataset.
# In order to merge all the CSV files into a single dataframe,
# make sure all the CSVs are stored in a sinlge folder and then navigate to that folder

# Importing libraries
import pandas as pd, numpy as np
import os, glob

# Importing all the CSVs
dataset = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "*.csv"))))
dataset = dataset.sample(frac = 1).reset_index(drop = True)
dataset.dropna(axis = 0, inplace = True)

# Dividing into independent feature matrix & dependent variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling - Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into Training & Testing subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Fitting the classifier on training dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 40, random_state = 1)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# Evaluating model performance through K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())

