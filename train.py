import pandas as pd
import numpy as np
import pickle
import sklearn.ensemble as ske
from sklearn import tree, linear_model, svm
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

indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    print("%d. feature %s (%f)" % (
        f + 1, data.columns[2 + indices[f]], fsel.feature_importances_[indices[f]]))

# Take care of the feature order
for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(data.columns[2 + f])

# Compare algorithms
algorithms = {
    "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=100, learning_rate=.05),
    "RandomForest": ske.RandomForestClassifier(n_estimators=100),
    "CVS": svm.SVC(),
    "SGD": linear_model.SGDClassifier(),
    "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
    "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
    "GNB": GaussianNB()
}

results = {}
print("\nTesting algorithms...")
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f %%" % (algo, score * 100))
    results[algo] = score

winner = max(results, key=results.get)
print('\nWinner algorithm is %s with a %f %% success' %
      (winner, results[winner] * 100))

# Save the algorithm and the feature list for later predictions
print('Saving algorithm and feature list in classifier directory...')
joblib.dump(algorithms[winner], 'classifier/classifier.pkl')
open('classifier/features.pkl', 'wb').write(pickle.dumps(features))
print('Saved...')
