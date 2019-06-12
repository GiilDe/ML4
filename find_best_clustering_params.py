from get_prepared_data import get_prepared_data
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)

X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])


best_initialization = None
best_score = float('-inf')
for _ in range(10):
    initialization = np.random.normal(0.5, 1, len(train_X.columns))
    kmean = KMeans(2, initialization)
    score = cross_val_score(kmean, X_to_split, Y_to_split, scoring='completeness_score')
    if score > best_score:
        best_score = score
        best_initialization = initialization


kmean = KMeans(2, best_initialization)
kmean.fit(X_to_split, Y_to_split)
clusters = kmean.predict(test_X)

