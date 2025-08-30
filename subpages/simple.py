#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/8/29 15:40
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   simple.py
# @Desc     :   

from numpy import linspace
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from streamlit import (empty, sidebar, subheader, session_state, button,
                       columns, metric, rerun, slider, caption)

from utils.helper import Timer, scatter_category

empty_messages: empty = empty()
left, right = columns(2, gap="large")
empty_chart: empty = empty()
empty_formula_title: empty = empty()
empty_formula: empty = empty()
empty_table_title: empty = empty()
empty_table: empty = empty()

if "data" not in session_state:
    session_state["data"] = None
if "passed" not in session_state:
    session_state["passed"] = None
if "y" not in session_state:
    session_state["y"] = None
if "timer_s" not in session_state:
    session_state["timer_s"] = None
if "model" not in session_state:
    session_state["model"] = None

with sidebar:
    if session_state["data"] is None:
        empty_messages.error("No data available. Please generate data in the 'Data Preparation' section first.")
    else:
        empty_messages.success("Data is available. You can proceed with Logistic Regression.")

        X = session_state["data"].drop(columns=[session_state["passed"]])
        Y = session_state["data"][session_state["passed"]]

        if session_state["y"] is None:
            subheader("Training Settings")
            if button("Train Logistic Regression Model", type="primary", use_container_width=True):
                with Timer(description="Logistic Regression Training") as t:
                    # Train the Logistic Regression model
                    session_state["model"] = LogisticRegression()
                    session_state["model"].fit(X, Y)

                    # Make predictions
                    session_state.y = session_state["model"].predict(X)
                    session_state["timer_s"] = t
                    rerun()
        else:
            # comparison: DataFrame = DataFrame({"Actual": Y, "Predicted": session_state.y})
            # empty_table_title.markdown("**The Results of the Trained Logistic Regression**")
            # empty_table.data_editor(comparison, hide_index=True, disabled=True, use_container_width=True)

            accuracy: float = accuracy_score(Y, session_state.y)
            percentage: float = round(accuracy * 100, 1)
            with left:
                metric("Accuracy", f"{percentage} %", delta=f"{percentage - 100} %", delta_color="normal")
            empty_messages.success(f"Logistic Regression Training Complete. {session_state.timer_s}")

            theta_0: float = session_state["model"].intercept_[0]
            theta_1: float = session_state["model"].coef_[0][0]
            theta_2: float = session_state["model"].coef_[0][1]
            x_1: float = session_state["data"].iloc[:, 0]

            empty_formula_title.markdown("**The Formula of the Trained Logistic Regression**")
            empty_formula.latex(
                r"""
                \text{Logit}(p) = \theta_0 + \theta_1 \cdot X_1 + \theta_2 \cdot X_2 \\
                p = \frac{1}{1 + e^{-(\theta_0 + \theta_1 \cdot X_1 + \theta_2 \cdot X_2)}}
                """
                .replace(r"\theta_0", f"{theta_0:.4f}")
                .replace(r"\theta_1", f"{theta_1:.4f}")
                .replace(r"\theta_2", f"{theta_2:.4f}")
            )

            x_2 = -(theta_0 + theta_1 * x_1) / theta_2
            fig = scatter_category(
                session_state.data,
                session_state["data"].columns[0], session_state["data"].columns[1],
                session_state.passed
            )
            fig.add_scatter(
                x=x_1,
                y=x_2,
                mode="lines",
                name="Category Boundary",
                line=dict(color="red", dash="dash", width=3)
            )
            empty_chart.plotly_chart(fig, use_container_width=True)

            subheader("Test Settings")
            mock_i: int = slider(
                "Score of Mock Exam I",
                min_value=1,
                max_value=100,
                value=int(session_state["data"].iloc[:, 0].mean()),
                step=1,
                help="Select a random score of the Mock Exam I"
            )
            caption(f"The score of Mock Exam I is {mock_i}.")

            mock_ii: int = slider(
                "Score of Mock Exam II",
                min_value=1,
                max_value=100,
                value=int(session_state["data"].iloc[:, 1].mean()),
                step=1,
                help="Select a random score of the Mock Exam II"
            )
            caption(f"The score of Mock Exam II is {mock_ii}.")

            if button("Prediction the Result of the Mock Exam III", type="primary", use_container_width=True):
                prediction = session_state["model"].predict(
                    DataFrame([[mock_i, mock_ii]], columns=["mock_exam_i", "mock_exam_ii"])
                )
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
