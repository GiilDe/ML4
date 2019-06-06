from get_prepared_data import get_prepared_data
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def get_similarities(parties_num, get_similarity, cluster_labels, data_X: pd.DataFrame):
    parties_similarities = {}
    for i in range(parties_num):
        for j in range(i+1, parties_num):
            parties_similarities[frozenset([i, j])] = get_similarity(i, j, cluster_labels, data_X)
    return parties_similarities


def get_party_in_cluster_num(party, data_X, label_num):
    return len(data_X[(data_X['Vote'] == party) & (data_X['cluster_label'] == label_num)])


def get_similarity(party1, party2, cluster_labels, df: pd.DataFrame):
    N = len(np.unique(cluster_labels))
    df = df.reset_index()
    cluster_labels = pd.Series(cluster_labels)
    df['cluster_label'] = cluster_labels
    party1_cluster_nums = np.array([get_party_in_cluster_num(party1, df, cluster) for cluster in range(N)])
    party2_cluster_nums = np.array([get_party_in_cluster_num(party2, df, cluster) for cluster in range(N)])

    normalized1 = normalize(party1_cluster_nums)
    normalized2 = normalize(party2_cluster_nums)
    difference = np.linalg.norm(normalized1-normalized2, ord=1)
    if difference == 0:
        difference = 1
    similarity = 1/difference

    return similarity


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()
cluster_res = KMeans(n_clusters=16).fit(train_X)
cluster_labels = cluster_res.labels_
train_X.insert(0, 'Vote', train_Y)

parties_num = len(np.unique(train_X['Vote']))
similarities = get_similarities(parties_num, get_similarity, cluster_labels, train_X)
