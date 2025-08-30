<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This application is designed to provide an intuitive and hands-on understanding of Logistic Regression, a fundamental
machine learning algorithm for binary classification problems. Unlike what its name suggests, it is used to predict
categorical outcomes (e.g., Yes/No, Spam/Not Spam, 1/0) rather than continuous values.

**FEATURES**
---

1. **Data Generation & Simulation**
    - Generate synthetic student exam data with a configurable random seed and data amount.
    - Automatically label pass/fail results.
2. **Interactive Data Exploration**
    - View data in editable tables (read-only mode).
    - Explore data trends with line charts and scatter plots.
    - Select feature columns and target columns for analysis.
3. **Simple Logistic Regression**
    - Train a standard logistic regression model.
    - Display linear decision boundary and formula.
    - Evaluate model accuracy.
    - Predict pass/fail for new exam scores.
4. **Polynomial Logistic Regression (Degree 2)**
    - Capture non-linear relationships with polynomial feature expansion.
    - Show symbolic and numeric quadratic decision boundaries.
    - Visualise non-linear decision boundaries.
    - Compare performance with simple logistic regression.
5. **Utility Functions & Reproducibility**
    - Timer to measure code execution time.
    - SeedSetter to control random seed for reproducibility.
    - Helper functions for data generation, plotting, and boundary calculation.
6. **Streamlit Interactive Interface**
    - Sidebar controls for data generation, feature selection, and model training.
    - Responsive charts and metrics for instant feedback.
    - Explore linear vs polynomial decision boundaries interactively.

**WEB DEVELOPMENT**
---

1. Install NiceGUI with the command `pip install streamlit`.
2. Run the command `pip show streamlit` or `pip show streamlit | grep Version` to check whether the package has been
   installed and its version.

**PRIVACY NOTICE**
---
This application may require inputting personal information or private data to generate customised suggestions,
recommendations, and necessary results. However, please rest assured that the application does **NOT** collect, store,
or transmit your personal information. All processing occurs locally in the browser or runtime environment, and **NO**
data is sent to any external server or third-party service. The entire codebase is open and transparent — you are
welcome to review the code [here](./) at any time to verify how your data is handled.

**LICENCE**
---
This application is licensed under the [BSD-3-Clause License](LICENSE). You can click the link to read the licence.

**CHANGELOG**
---
This guide outlines the steps to automatically generate and maintain a project changelog using git-changelog.

1. Install the required dependencies with the command `pip install git-changelog`.
2. Run the command `pip show git-changelog` or `pip show git-changelog | grep Version` to check whether the changelog
   package has been installed and its version.
3. Prepare the configuration file of `pyproject.toml` at the root of the file.
4. The changelog style is [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. Run the command `git-changelog`, creating the `Changelog.md` file.
6. Add the file `Changelog.md` to version control with the command `git add Changelog.md` or using the UI interface.
7. Run the command `git-changelog --output CHANGELOG.md` committing the changes and updating the changelog.
8. Push the changes to the remote repository with the command `git push origin main` or using the UI interface.
