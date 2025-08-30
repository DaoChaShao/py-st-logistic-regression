#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/8/30 14:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   polynomial.py
# @Desc     :

from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from streamlit import (empty, sidebar, subheader, session_state, button,
                       rerun, columns, metric, slider, caption)

from utils.helper import Timer, scatter_category, quadratic_decision_boundary_getter

empty_messages: empty = empty()
left, mid, right = columns(3, gap="large")
empty_table: empty = empty()
empty_chart: empty = empty()
empty_formula_symbolic: empty = empty()
empty_formula_numeric: empty = empty()

if "data" not in session_state:
    session_state["data"] = None
if "polynomial" not in session_state:
    session_state["polynomial"] = None
if "passed" not in session_state:
    session_state["passed"] = None
if "timer_ploy" not in session_state:
    session_state["timer_ploy"] = None
if "percentage" not in session_state:
    session_state["percentage"] = None

with (sidebar):
    if session_state["data"] is None:
        empty_messages.error("No data available. Please generate data in the 'Data Preparation' section first.")
    else:
        empty_messages.success("Data is available. You can proceed with Polynomial Logistic Regression.")

        X = session_state["data"].drop(columns=[session_state["passed"]])
        # Initialise the Polynomial Features transformer
        ploy = PolynomialFeatures(degree=2, include_bias=False)
        X = ploy.fit_transform(X)
        Y = session_state["data"][session_state["passed"]]

        if session_state["polynomial"] is None:
            subheader("Training Settings")
            empty_table.data_editor(X, hide_index=True, disabled=True, use_container_width=True)

            if button("Train Polynomial Logistic Regression Model", type="primary", use_container_width=True):
                with Timer("Train the Polynomial Logistic Regression Model") as t:
                    # Train the Logistic Regression model
                    session_state["polynomial"] = LogisticRegression(max_iter=1000)
                    session_state["polynomial"].fit(X, Y)
                    session_state["timer_ploy"] = t
                    rerun()
        else:
            empty_messages.success(f"{session_state.timer_ploy} Model trained successfully!.")

            y = session_state["polynomial"].predict(X)
            accuracy = accuracy_score(Y, y)
            percentage = round(accuracy * 100, 1)
            with left:
                metric("Polynomial Accuracy", f"{percentage} %", delta=f"{percentage - 100} %", delta_color="normal")
            if session_state["percentage"] is not None:
                with mid:
                    metric(
                        "Simple Accuracy",
                        f"{session_state.percentage} %", delta=f"{session_state.percentage - 100} %",
                        delta_color="normal"
                    )

            # Draw the decision boundary
            theta_0 = session_state["polynomial"].intercept_
            theta_1 = session_state["polynomial"].coef_[0][0]
            theta_2 = session_state["polynomial"].coef_[0][1]
            theta_3 = session_state["polynomial"].coef_[0][2]
            theta_4 = session_state["polynomial"].coef_[0][3]
            theta_5 = session_state["polynomial"].coef_[0][4]
            empty_formula_symbolic.latex(
                r"""
                    (\theta_5) x_2^2 + (\theta_4 x_1 + \theta_2) x_2 + (\theta_3 x_1^2 + \theta_1 x_1 + \theta_0) = 0
                """
            )
            empty_formula_numeric.latex(
                r"""
                    (\theta_5) x_2^2 \;+\; (\theta_4 x_1 + \theta_2) x_2 \;+\; (\theta_3 x_1^2 + \theta_1 x_1 + \theta_0) \;=\; 0
                """
                .replace(r"\theta_0", f"{theta_0[0]:.4f}")
                .replace(r"\theta_1", f"{theta_1:.4f}")
                .replace(r"\theta_2", f"{theta_2:.4f}")
                .replace(r"\theta_3", f"{theta_3:.4f}")
                .replace(r"\theta_4", f"{theta_4:.4f}")
                .replace(r"\theta_5", f"{theta_5:.4f}")
            )
            # print(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5)
            x_1 = session_state["data"].iloc[:, 0]
            x_1 = x_1.sort_values()
            a = theta_5
            b = theta_4 * x_1 + theta_2
            c = theta_3 * x_1 ** 2 + theta_1 * x_1 + theta_0
            discriminant = b ** 2 - 4 * a * c

            boundary_positive = (-b + discriminant ** 0.5) / (2 * a)
            boundary_negative = (-b - discriminant ** 0.5) / (2 * a)
            # print(boundary_positive, boundary_negative)

            # Using the helper function to get the decision boundary
            # poses = []
            # negs = []
            # for x in x_1:
            #     pos, neg = quadratic_decision_boundary_getter(session_state["polynomial"], x)
            #     poses.append(pos)
            #     negs.append(neg)

            fig = scatter_category(
                session_state["data"],
                x_name=session_state["data"].columns[0],
                y_name=session_state["data"].columns[1],
                category=session_state["passed"]
            )
            fig.add_scatter(
                x=x_1,
                y=boundary_positive,
                # y=poses,
                mode="lines", name="pos boundary",
                line=dict(color="red", dash="dash", width=3)
            )
            fig.add_scatter(
                x=x_1,
                y=boundary_negative,
                # y=negs,
                mode="lines", name="neg boundary",
                line=dict(color="red", dash="solid", width=3)
            )
            empty_chart.plotly_chart(fig, use_container_width=True)

            subheader("Test Settings")
            mock_i: int = slider(
                "Score of Mock Exam I",
                min_value=1,
                max_value=100,
                step=1,
                help="Select a random score of the Mock Exam I"
            )
            caption(f"The score of Mock Exam I is {mock_i}.")

            mock_ii: int = slider(
                "Score of Mock Exam II",
                min_value=1,
                max_value=100,
                step=1,
                help="Select a random score of the Mock Exam II"
            )
            caption(f"The score of Mock Exam II is {mock_ii}.")

            if button("Prediction the Result of the Mock Exam III", type="primary", use_container_width=True):

                entries: DataFrame = DataFrame([[mock_i, mock_ii]], columns=["mock_exam_i", "mock_exam_ii"])
                x = ploy.fit_transform(entries)

                prediction = session_state["polynomial"].predict(x)
                match prediction:
                    case "1":
                        with right:
                            metric("Prediction", "Pass", delta="👍", delta_color="normal")
                        empty_messages.success("The student is predicted to pass the exam.")
                    case "0":
                        with right:
                            metric("Prediction", "Fail", delta="👎", delta_color="inverse")
                        empty_messages.warning("The student is predicted to fail the exam.")
                    case _:
                        empty_messages.error("Prediction is inconclusive.")
