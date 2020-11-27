import pandas as pd
import numpy as np
import pickle
import sklearn.ensemble as ske
from sklearn import tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


# Read data.csv file
data = pd.read_csv('dataset.csv', sep=',')
X = data.drop(['ID', 'target'], axis=1).values
y = data['target'].values

print('Searching important feature based on %i total features\n' % X.shape[1])

# Feature selection using Trees Classifier
fsel = ske.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.2)

features = []

print('%i features identified as important:' % nb_features)


