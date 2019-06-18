from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd
from get_prepared_data import get_prepared_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from build_coalition_from_similarity_matrix import build_coalition_from_similarity_matrix


class clf_similarity():
    def __init__(self, clf):
        self.clf = clf

    def fit(self, train_X, train_Y):
        self.clf.fit(train_X, train_Y)

    def get_similarity(self, test_X, test_Y):
        labels = list(set(test_Y))
        X_by_label = dict()
        for label in labels:
            X_by_label[label] = test_X[test_Y == label]

        similarity_matrix = np.zeros((len(labels), len(labels)))

        for label_index in range(len(labels)):
            for x in self.clf.predict_proba(X_by_label[label_index]):
                for label_index_ in range(len(labels)):
                    similarity_matrix[label_index][label_index_] += x[label_index_]
            similarity_matrix[label_index] = similarity_matrix[label_index]/len(X_by_label[label_index])

        return similarity_matrix


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)
X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])
# cross k validation:
best_score = float('-inf')
best_clf = None
for clf in [GaussianNB(), BernoulliNB(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]:
    score = np.average(cross_val_score(clf, X_to_split, Y_to_split, cv=3))
    print(score)
    if score > best_score:
        best_score = score
        best_clf = clf

print(best_clf)

clf_similarity = clf_similarity(best_clf)
clf_similarity.fit(train_X, train_Y)
similarity_matrix = clf_similarity.get_similarity(X_to_split, Y_to_split)

plt.imshow(similarity_matrix)
plt.colorbar()
plt.show()




print(build_coalition_from_similarity_matrix(similarity_matrix, X_to_split, Y_to_split, test_X, test_Y))
