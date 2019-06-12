from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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


from get_prepared_data import get_prepared_data

train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)
clf_similarity = clf_similarity(GaussianNB())
clf_similarity.fit(train_X, train_Y)
plt.matshow(clf_similarity.get_similarity(validation_X, validation_Y))



plt.show()

