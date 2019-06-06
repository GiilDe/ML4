import pandas as pd
from FeatureSelection import relief, sfs
from pandas import DataFrame, Series
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def X_Y_2_XY(data_X: pd.DataFrame, data_Y: pd.DataFrame):
    X_columns_list = data_X.columns.values
    data_XY = data_X.copy()
    data_Y = pd.DataFrame(data_Y.copy())
    data_Y = data_Y.set_index(data_X.index)
    data_XY = data_XY.assign(Vote=data_Y)
    new_columns_list = ['Vote']
    new_columns_list.extend(X_columns_list)
    data_XY = data_XY[new_columns_list]
    return data_XY


def XY_2_X_Y(data_XY):
    return data_XY.iloc[:, 1:], data_XY.iloc[:, 0]


def split_data(all_data):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    train_index, test_valid_index = next(sss.split(all_data.iloc[:, 1:], all_data.iloc[:, 0]))
    train_X = all_data.iloc[train_index, 1:]
    train_Y = all_data.iloc[train_index, 0]

    test_valid_X = all_data.iloc[test_valid_index, 1:]
    test_valid_Y = all_data.iloc[test_valid_index, 0]
    test_valid_XY = all_data.iloc[test_valid_index, :]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
    validation_index, test_index = next(sss.split(test_valid_X, test_valid_Y))

    validation_X = test_valid_XY.iloc[validation_index, 1:]
    validation_Y = test_valid_XY.iloc[validation_index, 0]
    test_X = test_valid_XY.iloc[test_index, 1:]
    test_Y = test_valid_XY.iloc[test_index, 0]

    return train_X, train_Y, validation_X, validation_Y, test_X, test_Y


def get_binary_features(df: DataFrame):
    binary_features = []
    for feature in df:
        if len(get_series_hist(df[feature])) == 2:
            binary_features.append(feature)
    return binary_features


def get_series_hist(series: Series):
    values = set()
    for value in series:
        if value is not np.nan:
            values.add(value)
    return values



# bad sample is a sample with nan in a categorial values
def count_bad_samples(df: DataFrame):
    categorials = ['Main_transportation', 'Occupation', 'Most_Important_Issue']
    bad = 0
    for _, sample in df.iterrows():
        cat_values = set(sample[feature] for feature in categorials)
        if np.nan in cat_values:
            bad += 1
    return bad


def remove_bad_samples(df: DataFrame):
    categorials = ['Main_transportation', 'Occupation', 'Most_Important_Issue']
    for i, sample in df.iterrows():
        cat_values = set(sample[feature] for feature in categorials)
        if np.nan in cat_values:
            df.drop(axis=0, index=i)


def to_numerical_data(data: DataFrame):
    # remove_bad_samples(data)
    # convert_binary(data)
    data_featues_one_hot = data.drop(columns='Vote')
    data_featues_one_hot = pd.get_dummies(data_featues_one_hot)
    data_featues_one_hot.insert(0, 'Vote', data['Vote'])
    data_featues_one_hot['Vote'] = data_featues_one_hot['Vote'].map({
        'Khakis': 0, 'Oranges': 1, 'Purples': 2, 'Turquoises': 3, 'Yellows': 4, 'Blues': 5, 'Whites': 6,
        'Greens': 7, 'Violets': 8, 'Browns': 9, 'Greys': 10, 'Pinks': 11, 'Reds': 12,
    })
    return data_featues_one_hot


def chosen_features(data: DataFrame):
    np_data = data.to_numpy()
    clf = DecisionTreeClassifier()
    sfs_chosen_features = sfs(clf, data, np_data[0:8000, :], np_data[8000:, :], 30)
    relief_chosen_features = relief(data, np_data, threshold=0.3, times=3)
    chosen_features = sfs_chosen_features.intersection(relief_chosen_features)
    return chosen_features


class Scaler:
    def __init__(self, train_XY):
        features_to_normalize = ['Yearly_ExpensesK', 'Weighted_education_rank', 'Number_of_valued_Kneset_members']
        features_to_standartize = ['Avg_environmental_importance', 'Avg_government_satisfaction',
                                   'Avg_education_importance', 'Avg_monthly_expense_on_pets_or_plants',
                                   'Avg_Residancy_Altitude']

        self.features_to_normalize = features_to_normalize
        self.features_to_standartize = features_to_standartize
        self.dict = {}

    def fit(self, train_XY):
        for feature in (self.features_to_normalize + self.features_to_standartize):
            description = train_XY[feature].describe()
            if feature in self.features_to_standartize:
                self.dict.update({feature: (description.mean(), description.std())})
            else:
                assert feature in self.features_to_normalize
                self.dict.update({feature: (description.max(), description.min())})

    def scale(self, data: DataFrame):
        for feature in self.features_to_normalize:
            self.normalize(feature, data)
        for feature in self.features_to_standartize:
            self.standartize(feature, data)

    def standartize(self, feature, data: DataFrame):
        dict_val = self.dict.get(feature)
        mean = dict_val[0]
        std = dict_val[1]
        if std == 0:
            std = 0.00001
        data[feature] = (data[feature] - mean)/std

    def normalize(self, feature, data: DataFrame):
        dict_val = self.dict.get(feature)
        max = dict_val[0]
        min = dict_val[1]
        if min == max:
            data[feature] = (data[feature] - min)
        else:
            data[feature] = (data[feature] - min) / (max - min)


def plot_features_hists(data: DataFrame):
    for i, column in enumerate(data):
        name = str(i) + ' ' + column
        data[column].plot(kind='hist')
        plt.title(name)
        plt.show()


def plot_scatters(data: DataFrame):
    for i, column in enumerate(data):
        if i != 0:
            name = str(i) + ' ' + 'vote and ' + column
            plt.title(name)
            plt.scatter(data[column], data['Vote'])
            plt.show()


def count(data: DataFrame, feature_name):
    no = np.zeros(13)
    yes = np.zeros(13)
    for _, sample in data.iterrows():
        n = int(sample['Vote'])
        if sample[feature_name] == 0:
            no[n] += 1
        else:
            yes[n] += 1

    y = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    plt.scatter(no, y)
    plt.scatter(yes, y)
    plt.title(feature_name)
    plt.show()


def plot_vote_to_features_colored(data: DataFrame):
    names = data.columns.values
    for i in range(1, 52):
        sns.pairplot(data.iloc[:, [0, i]], hue='Vote')
        name = 'Vote to ' + str(names[i])
        plt.title(name)
        plt.savefig(name + '.png')
        plt.show()


def plot_vote_to_features(data: DataFrame):
    names = data.columns.values
    for i in range(1, 52):
        for j in range(0, 12):
            data_labeled = data[data.Vote == j]
            sns.pairplot(data_labeled.iloc[:, [i]])
            name = 'Vote labeled ' + str(j) + ' to ' + str(names[i])
            plt.title(name)
            #plt.show()
            plt.savefig(name + '.png')


# def arrange_data(df: DataFrame):
#     features_to_normalize = [1, 10, 11, 12, 13, 14, 16, 17, 25, 27, 30, 33]
#     features_to_standartize = [2, 3, 4, 6, 15, 18, 19, 20, 22, 23, 24, 26, 29, 31, 32]
#     data_featues_one_hot = to_numerical_data(df)
#     data = data_featues_one_hot.fillna(method='ffill')
#     normalize_names = [data.columns.values[i] for i in features_to_normalize]
#     standartize_names = [data.columns.values[i] for i in features_to_standartize]
#     scale(data, normalize_names, standartize_names)
#     return data


def scale_all(train_XY, validation_XY, test_XY):
    scaler = Scaler(train_XY)
    scaler.fit(train_XY)
    scaler.scale(train_XY)
    scaler.scale(validation_XY)
    scaler.scale(test_XY)
    return train_XY, validation_XY, test_XY


def sklearn_scale_all(train_XY, validation_XY, test_XY):
    scaler = StandardScaler()
    scaler.fit(train_XY)
    scaler.transform(train_XY)
    scaler.transform(validation_XY)
    scaler.transform(test_XY)
    return train_XY, validation_XY, test_XY