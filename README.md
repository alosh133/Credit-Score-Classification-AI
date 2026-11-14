# 💎 FinScore AI — Credit Score Classification System

**Course Final Project | Epsilon AI Academy | Developed by Ali Hamed**

## 📌 Overview
**FinScore AI** is an end-to-end Machine Learning classification project and interactive web application designed to evaluate customer creditworthiness (**Good**, **Standard**, **Poor**) based on global **FICO** credit scoring standards.

## 🚀 Live Demo
🔗 **[Open FinScore AI Web App](https://credit-score-classification-ai-rxu7prfigvdwuyglbluzud.streamlit.app/)**

## 🧠 Machine Learning Pipeline
* **Dataset:** Credit Score Classification Dataset (100,000 records, 28 features) — [Kaggle Dataset Link](https://kaggle.com).
* **Data Cleaning & Leak Prevention:** Handled missing values, outliers, and prevented data leakage by splitting data by `Customer_ID` using `GroupShuffleSplit`.
* **Feature Engineering:** Created domain-specific indicators including `Debt_to_Income_Ratio` and `Payment_History_Risk`.
* **Model Selection & Tuning:** Evaluated 6 algorithms using `Macro F1-Score` and tuned **XGBoost** via `RandomizedSearchCV` with balanced class weights, achieving **82% Recall** on the minority `Good` class.

## 🛠️ Tech Stack
`Python` • `Scikit-Learn` • `XGBoost` • `Pandas` • `Streamlit` • `Plotly`
