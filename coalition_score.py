import numpy as np
import pandas as pd
from get_prepared_data import get_prepared_data
from itertools import chain, combinations


def coalition_score(coalition_, XY):
    coalition = XY[XY['Vote'].isin(coalition_)].to_numpy()
    opposition = XY[~XY['Vote'].isin(coalition_)].to_numpy()
    coalition = coalition[:, 1:]
    opposition = opposition[:, 1:]
    avg_coalition = np.average(coalition, axis=0)
    avg_opposition = np.average(opposition, axis=0)
    avg_dist = np.average([np.linalg.norm(x - avg_coalition, ord=2) for x in coalition])
    return np.linalg.norm(avg_coalition - avg_opposition, ord=2)/avg_dist**2  # np.linalg.norm(avg_coalition - avg_opposition, ord=2)/avg_dist

