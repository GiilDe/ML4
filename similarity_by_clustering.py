from get_prepared_data import get_prepared_data
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import KFold


def calc_sim(party1_cluster_hist, party2_cluster_hist):
    similarity, _ = sp.stats.pearsonr(party1_cluster_hist, party2_cluster_hist)
    return similarity


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
       return v
    return v / norm


def get_similarities(parties_num, get_similarity, cluster_labels, data_X: pd.DataFrame):
    parties_similarities = {}
    for i in range(parties_num):
        for j in range(i+1, parties_num):
            parties_similarities[frozenset([i, j])] = get_similarity(i, j, cluster_labels, data_X)
    return parties_similarities


def get_party_in_cluster_num(party, data, label_num, cluster_labels):
    if 'cluster_label' not in data.columns:
        data['cluster_label'] = cluster_labels

    return len(data[(data['Vote'] == party) & (data['cluster_label'] == label_num)])


def get_party_cluster_hist(party, cluster_labels, df: pd.DataFrame):
    N = len(np.unique(cluster_labels))
    party_cluster_hist = np.array([get_party_in_cluster_num(party, df, cluster, cluster_labels) for cluster in range(N)])
    return party_cluster_hist


def get_similarity(party1, party2, cluster_labels, df: pd.DataFrame):
    N = len(np.unique(cluster_labels))
    df = df.reset_index()
    cluster_labels = pd.Series(cluster_labels)
    df['cluster_label'] = cluster_labels

    party1_cluster_hist = np.array([get_party_in_cluster_num(party1, df, cluster) for cluster in range(N)])
    party2_cluster_hist = np.array([get_party_in_cluster_num(party2, df, cluster) for cluster in range(N)])

    similarity = calc_sim(party1_cluster_hist, party2_cluster_hist)

    return similarity


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(save=False)


# numerous clusters, look at similarities
k_means = KMeans(n_clusters=20)
cluster_res = k_means.fit(train_X)
cluster_labels = k_means.fitpredict(test_X)
test_X.insert(0, 'Vote', test_Y)

parties_num = len(np.unique(train_X['Vote']))
similarities = get_similarities(parties_num, get_similarity, cluster_labels, train_X)

