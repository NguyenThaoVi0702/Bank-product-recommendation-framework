# Predictive Analytics in Banking: Product Recommendation Engine

## Project Overview

This repository contains the complete research pipeline and a demonstration web application for the paper: *"Predictive analytics in banking: A paradigm of optimizing customer product recommendation with regression and iterative product augmentation"*.

The project is designed to solve a critical challenge in the banking sector: how to effectively recommend products to customers to maximize their Total Operating Income (TOI) for the bank, especially in resource-constrained or data-limited environments.

The core of this project is a machine learning pipeline that:
1.  Predicts a customer's potential TOI.
2.  Employs an **Iterative Product Augmentation** framework to identify the single "next best product" that will provide the highest financial lift.
3.  Is accompanied by a Flask web application to demonstrate how these recommendations can be served to a relationship manager in a practical setting.


## Key Features

-   **End-to-End Research Pipeline:** A single, executable script (`model_pipeline.py`) that reproduces the entire research workflow.
-   **Advanced Data Preprocessing:** Includes sophisticated techniques for handling duplicates, cleaning data, and removing multi-feature outliers.
-   **Robust Model Benchmarking:** Systematically compares five different regression models using six performance metrics, with performance optimizations for faster execution.
-   **Hyperparameter Optimization:** Utilizes `GridSearchCV` to fine-tune the best-performing model (LGBMRegressor) for maximum accuracy.
-   **GPU Acceleration:** Automatically leverages a GPU for model training if a compatible one is detected, significantly speeding up the process.
-   **Flask Web Demo:** A simple web application to visualize how a relationship manager would interact with the system to see customer information and product recommendations.

