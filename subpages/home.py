#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/8/28 19:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("Logistic Regression Example")
with expander("**INTRODUCTION**", expanded=True):
    caption("- Generate synthetic student exam data with configurable random seed and data amount.")
    caption("- Explore the generated data through interactive tables, line charts, and scatter plots.")
    caption("- Train a simple Logistic Regression model and visualize its linear decision boundary.")
    caption("- Train a Polynomial Logistic Regression model (degree 2) to capture non-linear decision boundaries.")
    caption("- Display both symbolic and numeric formulas for the polynomial decision boundary.")
    caption("- Evaluate model performance with accuracy metrics and compare simple vs polynomial models.")
    caption("- Test models by inputting new mock exam scores to predict pass/fail results.")
