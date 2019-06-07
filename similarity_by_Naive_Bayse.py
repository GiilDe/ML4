from sklearn.naive_bayes import GaussianNB

class Naive_Bayse_similarity():
    def __init__(self):
        self.NB = GaussianNB()

    def fit(self, train_X, train_Y):
        self.NB.fit(train_X, train_Y)

    def get_similarity(self, test_X, test_Y):
        labels = list(set(test_Y))
        X_by_label = dict()
        for label in labels:
            X_by_label[label] = test_X[test_Y['Vote'] == label]

        similarity_dict = dict()
        for label in labels:
            similarity_dict[label] = dict()

        for label in labels:
            for x in X_by_label[label]:
                self.NB.predict_proba(x)

from get_prepared_data import get_prepared_data

train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)
