#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/8/28 19:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   about.py
# @Desc     :

from streamlit import title, expander, caption

title("**Application Information**")
with expander("About this application", expanded=True):
    caption("- Provides a complete workflow from data generation to model training and testing.")
    caption("- Uses Streamlit for interactive UI and Plotly for data visualization.")
    caption("- Includes utility functions for timing code blocks, setting random seeds, and generating reproducible data.")
    caption("- Designed for educational purposes to demonstrate logistic regression concepts.")
    caption("- Supports hands-on exploration of linear vs polynomial decision boundaries in classification tasks.")
    caption("- Facilitates understanding of how feature combinations affect model predictions and boundaries.")
