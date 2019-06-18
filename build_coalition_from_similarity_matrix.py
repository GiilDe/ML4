import numpy as np
from coalition_score import coalition_score as f
from data_handling import X_Y_2_XY


def big_enough_coalition(parties_list, test_X, test_Y):
    coalition = test_X[test_Y.isin(parties_list)]
    return len(coalition) >= len(test_X) * 0.51


def most_similar_party(coalition, similarity_matrix):
    assert len(coalition) == len(set(coalition))
    similarity_list = np.zeros(13)
    for party in coalition:
        for other_party in set(range(13)) - set(coalition):
            similarity_list[other_party] += similarity_matrix[party][other_party] + similarity_matrix[other_party][party]

    return np.argmax(similarity_list)


def build_coalition_from_similarity_matrix(similarity_matrix, X_to_split, Y_to_split, test_X, test_Y):
    best_coalition_score = float('-inf')
    best_coalition = list(range(13))
    for party in range(13):
        coalition = [party]
        while not big_enough_coalition(coalition, test_X, test_Y):
            coalition.append(most_similar_party(coalition, similarity_matrix))
        coalition_score_ = f(coalition, X_Y_2_XY(X_to_split, Y_to_split))
        if coalition_score_ > best_coalition_score:
            best_coalition_score = coalition_score_
            best_coalition = coalition.copy()
        while len(coalition) < 13:
            coalition.append(most_similar_party(coalition, similarity_matrix))
            coalition_score_ = f(coalition, X_Y_2_XY(X_to_split, Y_to_split))
            if coalition_score_ > best_coalition_score:
                best_coalition_score = coalition_score_
                best_coalition = coalition.copy()

    best_coalition.sort()
    return best_coalition, best_coalition_score