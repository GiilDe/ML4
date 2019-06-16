import numpy as np
from get_prepared_data import get_prepared_data
train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)


def coalition_score(coalition_, XY):
    coalition = XY[XY['Vote'].isin(coalition_)].to_numpy()
    opposition = XY[~XY['Vote'].isin(coalition_)]
    avg_coalition = np.average(coalition, axis=0)
    avg_opposition = np.average(opposition, axis=0)
    avg_dist = np.average([np.linalg.norm(x - avg_coalition, ord=2) for x in coalition])
    return np.linalg.norm(avg_coalition - avg_opposition, ord=2)/avg_dist
