import pandas as pd
import numpy as np
from distribution_ import Distribution
from imputations import DistirbutionImputator
from sklearn.impute import SimpleImputer
from data_handling import split_data, X_Y_2_XY, XY_2_X_Y


class DistirbutionOutlinersCleaner:
    def __init__(self, nan_probability=0.5):
        self.nan_probability = nan_probability
        self.dict_with_respect_to_labels = dict()
        self.dict_without_respect_to_labels = dict()
        self.example_probabilities_list_WRTL = None
        self.example_probabilities_list_WN = None
        self.cell_probabilities_list_WRTL = None
        self.cell_probabilities_list_WN = None

    def fit(self, data: pd.DataFrame):
        """
        asserts first column is the label column.
        saves for every column it's mean and std, once with respect to label and once without.
        """

        data = data.copy()
        label_name = data.columns[0]
        data = data.sort_values(by=[str(label_name)])
        dividing_by_label = []
        current_label = data.iloc[0, 0]
        for i in range(len(data)):
            if data.iloc[i, 0] != current_label:
                dividing_by_label.append(i)
                current_label = data.iloc[i, 0]

        dividing_by_label.append(len(data))
        # print('dividing_by_label: ', dividing_by_label)


        for column_index in range(1, len(data.columns)):
            i = 0
            dist_WN = Distribution()
            dist_WN.Fit(data.iloc[:, column_index])
            self.dict_without_respect_to_labels.update({column_index: dist_WN})
            for j in dividing_by_label:
                dist_WRTL = Distribution()
                dist_WRTL.Fit(data.iloc[i:j, column_index])
                print(dist_WRTL.DistributionName)
                self.dict_with_respect_to_labels.update({(data.iloc[i, 0], column_index): dist_WRTL})
                i = j

        print('kililiiiiiii')
        # print('\n\nself.dict_without_respect_to_labels:\n', self.dict_without_respect_to_labels)
        # print('\nself.dict_with_respect_to_labels:\n', self.dict_with_respect_to_labels, '\n\n')

        #  WRTL is a shortcut to with_respect_to_label
        #  WN is a shortcut to with_no (respect_to_label)
        example_probabilities_list_WRTL = []
        cell_probabilities_list_WRTL = []
        cell_probabilities_list_WN = []
        example_probabilities_list_WN = []
        for i in range(len(data)):
            example_probability_WRTL = 1
            example_probability_WN = 1
            for j in range(1, len(data.columns)):
                feature = data.iloc[i, j]
                label = data.iloc[i, 0]
                dist_WRTL = self.dict_with_respect_to_labels.get((label, j))
                dist_WN = self.dict_without_respect_to_labels.get(j)
                if np.isnan(feature):
                    print('np.isnan(feature)')
                    example_probability_WRTL *= self.nan_probability * 25
                    example_probability_WN *= self.nan_probability * 25
                    cell_probability = self.nan_probability
                    cell_probabilities_list_WRTL.append((cell_probability, i, j, dist_WRTL))
                    cell_probabilities_list_WN.append((cell_probability, i, j, dist_WN))
                    continue
                #
                # std_WRTL = self.dict_with_respect_to_labels.get((label, j))[1]
                # if std_WRTL == 0:
                #     std_WRTL = 0.1
                # std_WN = self.dict_without_respect_to_labels.get(j)[1]
                #
                # if std_WN == 0:
                #     std_WN = 0.1
                # if np.isnan(dist_WRTL):
                #     mean_WRTL = mean_WN
                # if np.isnan(std_WRTL):
                #     std_WRTL = std_WN

                # if not np.isnan(dist_WRTL):
                cell_probability = dist_WRTL.pdf(feature)
                cell_probabilities_list_WRTL.append((cell_probability, i, j, dist_WRTL.Random()))
                example_probability_WRTL *= cell_probability * 25

                # if not np.isnan(dist_WN):
                cell_probability = dist_WN.pdf(feature)
                cell_probabilities_list_WN.append((cell_probability, i, j, dist_WN.Random()))
                example_probability_WN *= cell_probability * 25

            example_probabilities_list_WRTL.append((example_probability_WRTL, i))
            example_probabilities_list_WN.append((example_probability_WN, i))

        example_probabilities_list_WRTL.sort(key=lambda x: x[0])
        example_probabilities_list_WN.sort(key=lambda x: x[0])
        cell_probabilities_list_WRTL.sort(key=lambda x: x[0])
        cell_probabilities_list_WN.sort(key=lambda x: x[0])

        self.example_probabilities_list_WRTL = example_probabilities_list_WRTL
        self.example_probabilities_list_WN = example_probabilities_list_WN
        self.cell_probabilities_list_WRTL = cell_probabilities_list_WRTL
        self.cell_probabilities_list_WN = cell_probabilities_list_WN

    def clean_and_correct(self, data: pd.DataFrame, num_of_examples_to_delete, num_of_examples_to_correct, respect_to_labels=True):
        data = data.copy()
        if respect_to_labels:
            probabilities_list = self.example_probabilities_list_WRTL
        else:
            probabilities_list = self.example_probabilities_list_WN
        to_delete_list = [x[1] for x in probabilities_list[0:num_of_examples_to_delete]]

        if num_of_examples_to_correct > 0:
            if respect_to_labels:
                probabilities_list = [x for x in self.cell_probabilities_list_WRTL if x[1] not in to_delete_list]
            else:
                probabilities_list = [x for x in self.cell_probabilities_list_WN if x[1] not in to_delete_list]
            for k in range(num_of_examples_to_correct):
                to_correct = probabilities_list[k]
                i = to_correct[1]
                j = to_correct[2]
                data.iat[i, j] = to_correct[3]
        to_delete_list.sort()
        return data.drop(data.index[to_delete_list])


def clean_and_correct_train_X(train_XY):
    cleaner = DistirbutionOutlinersCleaner()
    cleaner.fit(train_XY)
    clean_correct_training_XY = cleaner.clean_and_correct(train_XY, 150, 450)
    return clean_correct_training_XY


def outliner_cleaner_test(data):
    train_X, train_Y, validation_X, validation_Y, test_X, test_Y = split_data(data)
    train_XY = train_X.copy()
    train_XY.insert(loc=0, column='Vote', value=train_Y)
    imp = DistirbutionImputator()
    imp.fit(train_XY)
    imputed_train_XY = imp.fill_nans(train_XY, data_is_with_label_column=True)
    cleaner = DistirbutionOutlinersCleaner()
    cleaner.fit(imputed_train_XY)
    simple_imputer = SimpleImputer()
    best_acc = 0
    best_num_of_examples_to_delete = 0
    best_number_of_cells_to_correct = 0
    for num_of_examples_to_delete in range(0, 500, 25):
        for number_of_cells_to_correct in range(0, 500, 50):
            clean_correct_training_XY = cleaner.clean_and_correct(imputed_train_XY, num_of_examples_to_delete, number_of_cells_to_correct)
            clean_correct_training_X = clean_correct_training_XY.iloc[:, 1:]
            train_Y = clean_correct_training_XY.iloc[:, 0]
            simple_imputer.fit(clean_correct_training_X)
            imputed_validation_X = simple_imputer.transform(pd.DataFrame(validation_X))
            accuracy = test_data_quality(clean_correct_training_X, train_Y, imputed_validation_X, validation_Y)
            print('\naccuracy: ', accuracy, ' for num_of_examples_to_delete=', num_of_examples_to_delete, ' number_of_cells_to_correct=', number_of_cells_to_correct)
            if accuracy > best_acc:
                best_acc = accuracy
                best_num_of_examples_to_delete = num_of_examples_to_delete
                best_number_of_cells_to_correct = number_of_cells_to_correct
    print('best_num_of_examples_to_delete: ', best_num_of_examples_to_delete, ' best_number_of_cells_to_correct: ', best_number_of_cells_to_correct, ' best_acc: ', best_acc)
