# write your code here
import numpy as np
import pandas as pd


class GiniImpurity:
    """Gini Impurity Calculator"""

    def __init__(self, df: pd.DataFrame, cat_variable: str = "Survived"):
        """Constructor of the class

        Args:
            df (pd.DataFrame): a dataframe to calculate the Gini Impurity Value
            cat_variable (str): a categorical variable to refer to when calculating the Gini Impurity
        """
        self.__df = df.copy()
        self.__cat_variable = cat_variable

    def discard(self, col_name: str):
        """Discarding or removing a column from the dataframe

        Args:
            col_name (str): a column name to remove from the dataframe
        """
        if col_name in self.__df.columns:
            self.__df.drop(columns=col_name, inplace=True)

    def gini_index(self, col_name: str, value: any):
        """Calculate the gini index of each categorical values

        Args:
            col_name (str): a column name to calculate the individual gini index
            value (any): a value found in the column to calculate the individual gini index

        Returns:
            Calculated gini index of a given value from specified column
        """
        df_filtered = self.__df[self.__df[col_name] == value]  # filtering the dataframe
        value_counts = df_filtered[self.__cat_variable].value_counts()  # calculating the categorical values count

        return 1 - sum(
            np.square(value / df_filtered.shape[0]) for value in value_counts
        )

    def weighted_gini_index(self, col_name: str):
        """Calculates the weighted gini index of a column

        Args:
            col_name (str): a column name to calculate the weighted gini index

        Returns:
            calculated weighted gini index of a given column
        """
        value_counts = self.__df[col_name].value_counts()  # counting the number of different values in a column
        number_of_values = self.__df[col_name].count()  # getting the total number of elements in a column

        return sum(
            # a probability of a given value multiplied by its gini index value
            (value_counts[value] / number_of_values) * self.gini_index(col_name, value) for value in value_counts.index
        )

    def calculate(self, discard_col: str = None):
        """Loop over the columns of the dataframe and calculate the gini index

        Args:
            discard_col (str, optional): if there is a column that must be dropped before calculating the gini index

        Returns:
            a tuple of minimum weighted gini index value and splitting column if a dataframe is splittable
            if not None
        """
        if discard_col:
            self.discard(discard_col)

        # getting the columns to discard the categorical variable
        columns = self.__df.columns.to_list()
        columns.remove(self.__cat_variable)

        # if the dataframe is splittable
        # calculate the gini index, if not return None
        if len(columns):
            weighted_gini_values = {
                # calculate the weighted gini index for all columns
                col_name: self.weighted_gini_index(col_name) for col_name in columns
            }

            # find the minimum
            values = weighted_gini_values.values()
            minimum_gini = min(values)
            next_splitter = list(weighted_gini_values.keys())[list(values).index(minimum_gini)]

            return np.round(minimum_gini, 4), next_splitter

    def split(self, discard_col: str = None):
        """Split the dataframe after calculating the gini index

        Args:
            discard_col (str, optional): if there is a column that must be dropped before calculating the gini index

        Returns:
            a tuple of minimum weighted gini index value, splitting column name, and splits if a dataframe is splittable
            if not None
        """
        if calculated_gini := self.calculate(discard_col):
            gini_index, col_name = calculated_gini
            value_counts = self.__df[col_name].value_counts()
            chosen_value = value_counts.idxmax()

            # splitting the dataset
            splits = [
                list(self.__df[self.__df[col_name] == value].index)
                for value in value_counts.index
            ]

            # discarding the column after splitting
            self.discard(col_name)

            return gini_index, col_name, chosen_value, *splits


def runner():
    path_to_dataset = input()
    df = pd.read_csv(path_to_dataset, index_col=0)

    gini_impurity = GiniImpurity(df)
    split_result = gini_impurity.split()

    if split_result:
        print(*split_result)


if __name__ == "__main__":
    runner()
