#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/8/28 19:40
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :

from pandas import DataFrame
from streamlit import (empty, sidebar, subheader, number_input, slider,
                       button, session_state, rerun, selectbox, multiselect)

from utils.helper import Timer, SeedSetter, data_generator, scatter_category

empty_messages: empty = empty()
empty_scatter: empty = empty()
empty_table: empty = empty()
empty_line: empty = empty()

if "data" not in session_state:
    session_state["data"] = None
if "timer_p" not in session_state:
    session_state["timer_p"] = None
if "passed" not in session_state:
    session_state["passed"] = None

with sidebar:
    subheader("Data Settings")

    seed: int = number_input(
        "Random Seed",
        min_value=0, max_value=9999, value=9527, step=1,
        help="Set the random seed for reproducibility.",
    )
    amount: int = slider(
        "Amount of Data",
        min_value=100, max_value=200, value=100, step=1,
        help="Set the amount of data to be generated.",
    )

    if session_state["data"] is None:
        empty_messages.error("No data generated yet. Please generate data first.")

        if button("Generate Data", type="primary", use_container_width=True):
            with Timer(description="Data Generation") as t, SeedSetter(seed):
                session_state["data"] = data_generator(amount)
                empty_table.data_editor(
                    session_state["data"], hide_index=True, disabled=True, use_container_width=True
                )
            session_state["timer_p"] = t
            rerun()
    else:
        empty_table.data_editor(
            session_state["data"], hide_index=True, disabled=True, use_container_width=True
        )

        session_state.passed = selectbox(
            "Select the Pass/Fail Column",
            options=session_state["data"].columns.tolist(),
            index=2, disabled=True,
            help="Select the column that indicates whether the student passed or failed.",
        )

        cols: list[str] = [col for col in session_state["data"].columns.tolist() if col != session_state.passed]

        features: list = multiselect(
            "Select Feature Columns",
            options=cols,
            disabled=False,
            help="Select the feature columns to be used for prediction.",
        )
        if features:
            empty_messages.success("Feature columns selected. Displaying line chart.")
            with Timer(description="Displaying Line Chart") as t:
                empty_line.line_chart(
                    session_state["data"][features],
                    x_label="Index",
                    y_label="Value",
                    use_container_width=True,
                )
            empty_messages.success(f"Feature columns selected. {t}")
        else:
            empty_messages.warning(
                f"{session_state.timer_p} Please select at least one feature column to display the chart."
            )

        x_name: str = selectbox(
            "Select the Feature Column (X)",
            options=cols,
            help="Select the feature column as axis X.",
        )
        cols.remove(x_name)
        y_name = selectbox(
            "Select the Target Column (Y)",
            options=cols,
            help="Select the target column as axis Y.",
        )
        fig = scatter_category(session_state["data"], x_name, y_name, session_state.passed)
        empty_scatter.plotly_chart(fig, use_container_width=True)
