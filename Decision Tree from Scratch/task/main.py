import numpy as np
import pandas as pd


class GiniImpurity:
    """Gini Impurity Calculator"""

    def gini_index(self, classes: list):
        # calculating the categorical value counts
        return 1 - np.sum(np.square(np.unique(classes, return_counts=True)[1] / len(classes)))

    def weighted_gini_index(self, splits: list):
        # getting the total number of elements
        total_counts = sum(len(split) for split in splits)  # getting the total number of elements in a column

        # a probability of a given value multiplied by its gini index value
        return sum((len(classes) / total_counts) * self.gini_index(classes) for classes in splits)


class Node:

    def __init__(self):
        self.left = None
        self.right = None
        self.term = False
        self.label = None
        self.feature = None
        self.value = None

    def set_split(self, feature, value):
        self.feature = feature
        self.value = value

    def set_term(self, label):
        self.term = True
        self.label = label


class Split:
    def __init__(self, df: pd.DataFrame, cat_variable: str = "Survived", min_data_num: int = 1):
        self.__df = df.copy()
        self.__features = df.loc[:, df.columns != cat_variable].columns.tolist()
        self.__cat_variable = cat_variable
        self.__min_data_num = min_data_num
        self.__gini_impurity = GiniImpurity()

    def split(self, df: pd.DataFrame, classes: pd.Series):
        split_feature, split_value, left, right, global_min_gini = None, None, [], [], 1

        for feature, feature_data in df[self.__features].iteritems():
            for value in feature_data.unique():
                left_children = feature_data[feature_data == value].index.tolist()
                right_children = feature_data[feature_data != value].index.tolist()
                current_gini = self.__gini_impurity.weighted_gini_index(
                    [classes[left_children], classes[right_children]])

                if current_gini < global_min_gini:
                    global_min_gini = current_gini
                    split_feature = feature
                    split_value = value
                    left = left_children
                    right = right_children

        return global_min_gini, split_feature, split_value, left, right

    def are_all_features_equal(self, df: pd.DataFrame):
        return all(data.nunique() == 1 for _, data in df.iteritems())

    def is_stopping_criterion_fulfilled(self, df: pd.DataFrame, gini_impurity: float):
        return self.are_all_features_equal(df) or gini_impurity == 0 or df.shape[0] <= self.__min_data_num

    def recursive_split(self, node: Node, df: pd.DataFrame = None, classes: list = None):
        if df is None:
            classes = self.__df[self.__cat_variable]
            df = self.__df[self.__features]

        if self.is_stopping_criterion_fulfilled(df, self.__gini_impurity.gini_index(classes)):
            node.set_term(classes.value_counts().idxmax())
        else:
            w_gini, feature, val, left, right = self.split(df, classes)
            node.set_split(feature, val)
            print(f"Made split: {feature} is {val}")

            node.left = Node()
            self.recursive_split(node.left, df.loc[left, :], classes[left])

            node.right = Node()
            self.recursive_split(node.right, df.loc[right, :], classes[right])


def runner():
    path_to_dataset = input()
    df = pd.read_csv(path_to_dataset, index_col=0)

    root = Node()
    spliter = Split(df)
    spliter.recursive_split(root)


if __name__ == "__main__":
    runner()
