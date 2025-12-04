# 🌾 Wheat Seed Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/Library-TensorFlow-orange)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project is an end-to-end Machine Learning application designed to classify wheat seeds into three varieties (**Kama**, **Rosa**, and **Canadian**) based on their geometric properties.

Using the **UCI Seeds Dataset**, I developed a Feed-Forward Neural Network (ANN) that analyzes distinct physical features such as Area, Perimeter, and Compactness to automate the sorting process.

## 🚀 Live Demo
**[Click here to view the Live App]( INSERT_YOUR_STREAMLIT_LINK_HERE )**

## 📊 Model Performance
The model was trained on 210 samples using a 80/20 train-test split.
- **Architecture:** Multi-Layer Perceptron (MLP) with 2 Hidden Layers (64/32 units).
- **Accuracy:** ~86% on Test Data.
- **Key Metrics:** Achieved **100% Precision** on Kama seeds and **100% Recall** on Rosa/Canadian seeds.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow/Keras (Sequential API)
- **Data Processing:** Pandas, NumPy, Scikit-Learn (StandardScaler)
- **Visualization:** Seaborn, Matplotlib
- **Deployment:** Streamlit Cloud

## 📂 Project Structure