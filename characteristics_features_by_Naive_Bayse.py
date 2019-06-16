import pandas as pd
import numpy as np
from get_prepared_data import get_prepared_data
from sklearn.naive_bayes import GaussianNB


class characteristics_features_by_Naive_Bayse:
    def find_characteristics_features(self, X, Y):
        NB = GaussianNB()
        column_index = 0
        for column in X:
            NB.fit(X[column].to_numpy().reshape(-1, 1), Y)
            # min_val = column.min()
            # max_val = column.max()
            for val in np.linspace(X[column].min() * 1.1, X[column].max() * 0.9, 10):
                # print(NB.predict_proba(np.array([[avg], [-100000000]]))[0])
                probabilities = list(NB.predict_proba(np.array([[val], [0]]))[0])
                index = 0
                for proba in probabilities:
                    if proba > 0.4:
                        print(list(X)[column_index], ' with val ', '{:.2f}'.format(val), ' is a characteristics feature for party ', index, ' with probability: ', '{:.2f}'.format(proba))
                    index += 1
            column_index += 1


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)
cf_NB = characteristics_features_by_Naive_Bayse()
cf_NB.find_characteristics_features(train_X, train_Y)
