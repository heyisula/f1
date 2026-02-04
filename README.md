# Formula 1 Race Winner Prediction Model

A comprehensive machine learning project that predicts Formula 1 race winners using historical data from 1950 to 2024. This version includes a critical fix for data leakage, standardized feature scaling, and a modern web interface with intelligent auto-fill capabilities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)

## 📋 Table of Contents

- [Overview](#overview)
- [Critical Implementation Fixes](#critical-implementation-fixes)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Performance](#model-performance)
- [Web Application](#web-application)

## 🎯 Overview

This project trains and compares multiple machine learning algorithms to predict Formula 1 race winners. Unlike traditional models that might suffer from data leakage, this system uses **pre-race data** (cumulative points earned *before* the start) to provide realistic win probabilities.

### Key Highlights

- **Data Leakage Prevention**: Uses pre-race `cumulative_points` instead of post-race points.
- **Multi-Model Comparison**: Evaluates Random Forest, Gradient Boosting, XGBoost, and more.
- **Robust Preprocessing**: Standardized feature scaling and automated categorical encoding.
- **Intelligent UI**: Modern dark-mode interface with circuit-specific lap count auto-fill.
- **Advanced Evaluation**: Cross-validation (5-fold), Precision-Recall curves, and Feature Importance analysis.

## ⚠️ Critical Implementation Fixes

The model has been audited and fixed to resolve several critical issues found in traditional F1 predictors:

1. **Fixing Data Leakage**: Replaced post-race `points` with `cumulative_points` (points earned before the current race). This dropped artificially high accuracy (~95%) to a scientifically sound and realistic range.
2. **Feature Scaling**: Implemented `StandardScaler` to ensure features like `cumulative_points` (0-400+) don't overpower `grid_position` (1-20).
3. **Class Imbalance Handling**: Applied `class_weight='balanced'` and specialized scorers (`f1_weighted`) to handle the fact that only 1 driver wins out of 20+ per race.

## 🚀 Usage

### Step 1: Train the Model

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Execute training**: Open `train.ipynb` and run all cells, or use the command line:
   ```bash
   jupyter nbconvert --execute --to notebook --inplace train.ipynb
   ```
   This saves the best model and preprocessing objects to `out/models/f1_model_data.pkl`.

### Step 2: Launch the Web App

1. **Start server**: `python app.py`
2. **Open browser**: Navigate to `http://127.0.0.1:5000`
3. **Analyze**: Select a race scenario and get an instant win probability.

## 🔧 Features

| Feature | Description | Importance |
|---------|-------------|------------|
| **Grid Position** | Starting position on the grid | High |
| **Cumulative Points**| Points earned *before* this race | Medium-High |
| **Laps** | Race distance (auto-filled by UI) | Low-Medium |
| **Driver/Team/Circuit**| Categorical identifiers | High |
| **Driver Age** | Calculated age at race time | Low |

## 📊 Model Performance

After fixing data leakage and implementing proper validation:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Weighted F1-Score** | ~0.96 | Excellent overall performance |
| **Winner (Class 1) F1** | ~0.60 | Realistic for imbalanced race data |
| **ROC-AUC** | ~0.95 | Strong discriminative ability |
| **CV Stability** | ±0.02 | High generalization confidence |

> **Note**: These metrics represent the `Gradient Boosting` model, which currently out-performs others in our pipeline.

## 🌐 Web Application

### Features
- **Dynamic Auto-fill**: Select a circuit (e.g., Monaco) and the "Laps" field automatically updates (78).
- **Interactive Ring**: Visualized win probability with dynamic color coding.
- **Robust Validation**: Server-side checks ensure all inputs (grid 1-20, positive points) are valid.
- **Metadata API**: Fetches available drivers and teams dynamically from the trained model's encoders.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Ergast F1 API](http://ergast.com/mrd/) via [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
