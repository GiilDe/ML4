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
            party_index = 0
            for party in set(Y.to_numpy()):
                avg = X[column][Y == party].mean()
                # print(NB.predict_proba(np.array([[avg], [-100000000]]))[0])
                proba = NB.predict_proba(np.array([[avg], [0]]))[0][party_index]
                party_index += 1
                if proba >= 0.3:
                    print(list(X)[column_index] + ' is a characteristics features for party ', party)
            column_index += 1


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)
cf_NB = characteristics_features_by_Naive_Bayse()
cf_NB.find_characteristics_features(train_X, train_Y)
