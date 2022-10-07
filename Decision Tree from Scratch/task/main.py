# write your code here
import numpy as np


def gini(array: list):
    """Calculate the gini impurity for a given array

    Args:
        array: list

    Returns:
        float: the gini impurity value
    """
    array = np.array(array).astype(int)
    probability_1 = array.sum() / array.size
    probability_0 = 1 - probability_1

    return 1 - (np.square(probability_0) + np.square(probability_1))


def weighted_gini(split_one: list, split_two: list):
    """Calculate the weighted gini impurity

    Args:
        split_one(list): a split from a given dataset
        split_two(list): another split from a given dataset

    Returns:
        float: the value of weighted gini impurity value
    """
    first_size = len(split_one)
    second_size = len(split_two)
    total_size = first_size + second_size

    return (first_size / total_size * gini(split_one)) \
                    + (second_size / total_size * gini(split_two))


array = input().split()
split_one = input().split()
split_two = input().split()

print(np.round(gini(array), 2), np.round(weighted_gini(split_one, split_two), 2))
