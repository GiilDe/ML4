import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from get_prepared_data import get_prepared_data
from data_handling import X_Y_2_XY

def plot_vote_to_features_colored(data: pd.DataFrame):
    names = data.columns.values
    for i in range(1, len(names)):
        sns.pairplot(data.iloc[:, [0, i]], hue='Vote')
        name = 'Vote to ' + str(names[i])
        plt.title(name)
        plt.savefig(name + '.png')
        plt.show()


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)
plot_vote_to_features_colored(X_Y_2_XY(train_X, train_Y))
