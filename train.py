import pandas as pd
import numpy as np
import pickle
import sklearn.ensemble as ske
from sklearn import tree, linear_model, svm
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.model_selection import train_test_split
import joblib
from sklearn.naive_bayes import GaussianNB


# выбрать лучше RandomForest

# Read data.csv file
data = pd.read_csv('dataset.csv', sep=',')
# X = data.drop(['ID', 'target'], axis=1).values
# y = data['target'].values

# ==================================================================
number_of_features = 14
X = data.iloc[:, 1:15]  # drop 'ID' and 'target'
y = data.iloc[:, -1]    # features columns
fsel = SelectKBest(k=number_of_features).fit(X, y)
X_new = fsel.transform(X)
dfscores = pd.DataFrame(fsel.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
print('Searching important feature based on %i total features\n' % X.shape[1])
print('10 features identified as important:')
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(number_of_features, 'Score'))  # print 10 best features
features = [x[0]for x in featureScores.nlargest(number_of_features, 'Score')[['Specs']].to_numpy()]


# Train test split
np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, random_state=0)

# ==================================================================

# print('Searching important feature based on %i total features\n' % X.shape[1])
# # Feature selection using Trees Classifier
# fsel = ske.ExtraTreesClassifier().fit(X, y)
# model = SelectFromModel(fsel, prefit=True)
# X_new = model.transform(X)
# nb_features = X_new.shape[1]
#
#
# # Train test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_new, y, random_state=0)
#
# features = []
#
# print('%i features identified as important:' % nb_features)
#
# indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
# for f in range(nb_features):
#     print("%d. feature %s (%f)" % (
#         f + 1, data.columns[2 + indices[f]], fsel.feature_importances_[indices[f]]))
#
# # Take care of the feature order
# for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
#     features.append(data.columns[2 + f])

#elasticnet
# Compare algorithms
algorithms = {
    "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=14, learning_rate=0.1),
    "RandomForest": ske.RandomForestClassifier(n_estimators=14, criterion='entropy', ),
    "SVC": svm.SVC(),
    "SGD": linear_model.SGDClassifier(loss='modified_huber', penalty='l1', max_iter=22),
    #"DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
    "AdaBoost": ske.AdaBoostClassifier(n_estimators=14),
    "GNB": GaussianNB()
}

results = {}
print("\nTesting algorithms...")
for algo in algorithms:
    np.random.seed(0)
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
