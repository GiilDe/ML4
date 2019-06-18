from itertools import chain, combinations
import pandas as pd
from coalition_score import coalition_score
from data_handling import X_Y_2_XY
from get_prepared_data import get_prepared_data

def big_enough_coalition(parties_list):
    coalition = test_X[test_Y.isin(parties_list)]
    return len(coalition) >= len(test_X) * 0.51


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)
X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])

# print(big_enough_coalition([0, 1, 2, 3]))
# print(coalition_score([0, 1, 2, 3], X_Y_2_XY(test_X, test_Y)))

def best_option_ever():
    best_coalition_score = float('-inf')
    best_coalition = None
    # counter = 0
    for coalition in powerset(set(train_Y)):
        # print(counter)
        # counter += 1
        if not big_enough_coalition(coalition):
            continue
        else:
            score = coalition_score(coalition, X_Y_2_XY(X_to_split, Y_to_split))
            if score > best_coalition_score:
                best_coalition_score = score
                best_coalition = coalition
    return best_coalition, best_coalition_score

print(best_option_ever())

