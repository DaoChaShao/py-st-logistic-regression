#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/8/28 19:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :

from pandas import DataFrame
from plotly.express import scatter
from random import seed as random_seed, getstate as get_state, setstate as set_state
from random import randint
from time import perf_counter


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print()
        print("-" * 50)
        print(f"{self._description} has been started.")
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            print("-" * 50)
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT been started."


class SeedSetter(object):
    """ Set a random seed for reproducibility. """

    def __init__(self, seed: int = 9527):
        """ Initialise the RandomSeed class with a seed. """
        self._seed = seed
        self._state_random = None

    def __enter__(self):
        """ Enter the context manager and set the random seed. """
        # Store the current state of random and Faker
        self._state_random = get_state()
        # Set the random seed for reproducibility
        random_seed(self._seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Exit the context manager and reset the random seed. """
        # Reset the random and Faker states to their original values.
        set_state(self._state_random)
        return False

    def __str__(self):
        """ Return a string representation of the random seed. """
        return f"SeedSetter with seed {self._seed}"


def data_generator(amount: int) -> DataFrame:
    """ Generate synthetic data for demonstration purposes.
    :param amount: the amount of data to be generated
    :return: a DataFrame containing the generated data
    """
    data: list = []

    for _ in range(amount):
        mock_i: int = randint(30, 100)
        mock_ii: int = randint(30, 100)

        if mock_i >= 60 and mock_ii >= 60:
            data.append({
                "mock_exam_i": mock_i,
                "mock_exam_ii": mock_ii,
                "pass": "1",
            })
        else:
            data.append({
                "mock_exam_i": mock_i,
                "mock_exam_ii": mock_ii,
                "pass": "0",
            })

    return DataFrame(data)


def scatter_category(data: DataFrame, x_name: str, y_name: str, category: str):
    """ Get the unique categories in the target column.
    :param data: the DataFrame containing the data
    :param x_name: the name of the feature column (X)
    :param y_name: the name of the target column (Y)
    :param category: the name of the category column
    :return: a scatter plot with different colours and symbols for each category
    """
    return scatter(
        data,
        x=x_name,
        y=y_name,
        color=category,
        symbol=category,
        hover_data=[x_name, y_name, category]
    )


def quadratic_decision_boundary_getter(model, x) -> tuple:
    """ Get the decision boundary of a polynomial logistic regression model.
    :param model: the trained polynomial logistic regression model
    :param x: the value of the feature column (X)
    :return: the positive and negative decision boundaries
    """
    theta_0 = model.intercept_[0]
    theta_1 = model.coef_[0][0]  # x1
    theta_2 = model.coef_[0][1]  # x2
    theta_3 = model.coef_[0][2]  # x1^2
    theta_4 = model.coef_[0][3]  # x1*x2
    theta_5 = model.coef_[0][4]  # x2^2

    a = theta_5
    b = theta_4 * x + theta_2
    c = theta_3 * x ** 2 + theta_1 * x + theta_0
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return None, None

    boundary_pos = (-b + discriminant ** 0.5) / (2 * a)
    boundary_neg = (-b - discriminant ** 0.5) / (2 * a)
    return boundary_pos, boundary_neg
