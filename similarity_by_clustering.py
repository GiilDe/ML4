from get_prepared_data import get_prepared_data
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import KFold
from coalition_score import coalition_score as getCoalitionScore
from data_handling import X_Y_2_XY


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

    party1_cluster_hist = get_party_cluster_hist(party1, cluster_labels, df)
    party2_cluster_hist = get_party_cluster_hist(party2, cluster_labels, df)

    similarity = calc_sim(party1_cluster_hist, party2_cluster_hist)

    return similarity


def most_similar_parties(party, similarities):
    similarity_list = [(similarities[frozenset([party, other_party])], other_party)
                        for other_party in range(13) if other_party != party]
    similarity_list.sort(reverse=True, key=lambda x: x[0])
    for i in range(len(similarity_list)):
        yield similarity_list[i][1]


def big_enough_coalition(parties_list):
    coalition = test_X[test_Y.isin(parties_list)]
    return len(coalition) >= len(test_X) * 0.51


def build_coalition_from_similarity_matrix(similarities):
    # we will go through every party and will try to build a coalition with similar parties
    best_coalition_score = float('-inf')
    best_coalition = []
    for party in range(13):
        coalition = [party]
        most_similar_party_left = most_similar_parties(party, similarities)
        while not big_enough_coalition(coalition):
            coalition.append(next(most_similar_party_left))
        coalition_score_ = getCoalitionScore(coalition, X_Y_2_XY(test_X, test_Y))
        if coalition_score_ > best_coalition_score:
            best_coalition_score = coalition_score_
            best_coalition = coalition
    return best_coalition, best_coalition_score


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(save=False)

data_test = X_Y_2_XY(test_X, test_Y, False)
# best_score = float('-inf')
# best_coalition = None
# best_k = None
# for k in range(2, 50, 2):
#     k_means = KMeans(n_clusters=k)
#     cluster_res = k_means.fit(train_X)
#     cluster_labels = k_means.predict(test_X)
#     parties_num = len(np.unique(XY['Vote']))
#     similarities = get_similarities(parties_num, get_similarity, cluster_labels, XY)
#     coalition, score = build_coalition_from_similarity_matrix(similarities)
#     if score > best_score:
#         best_score = score
#         best_coalition = coalition
#         best_k = k


best_coalition = {5, 9, 0, 7, 3, 4, 2}
best_k = 18
best_party_to_add = None
best_score = float('-inf')
for party in set(range(13)) - set(best_coalition):
    new_coalition = best_coalition.union({party})
    score = getCoalitionScore(new_coalition, data_test)
    if score > best_score:
        best_score = score
        best_party_to_add = party


party_to_add_voters = data_test[data_test['Vote'].isin([best_party_to_add])]
party_to_add = party_to_add_voters.to_numpy()[:, 1:]
avg_party_to_add = np.average(party_to_add, axis=0)
coalition = data_test[data_test['Vote'].isin(best_coalition)].to_numpy()[:, 1:]
avg_coalition = np.average(coalition, axis=0)
dist = avg_coalition - avg_party_to_add
t = [0]

dist_s = pd.Series(np.append(t, dist))

mutated_data_test = data_test.copy()
for i, row in mutated_data_test.iterrows():
    if row['Vote'] == best_party_to_add:
        for j in range(len(dist_s)):
            row[j] += dist_s[j]

new_coalition = best_coalition | {best_party_to_add}
score = getCoalitionScore(new_coalition, mutated_data_test)
old_score = getCoalitionScore(best_coalition, data_test)
#we can see that old_score < score so we strenghtend our coalition and made it bigger