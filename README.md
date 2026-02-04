# Formula 1 Race Winner Prediction Model

A comprehensive machine learning project that predicts Formula 1 race winners using historical data from 1950 to 2024. This system compares multiple ML algorithms, automatically selects the best performer, and deploys it via a premium web application with a modern dark-mode interface.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![pandas](https://img.shields.io/badge/pandas-Latest-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-11557c.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Latest-blue.svg)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

## 📋 Table of Contents

- [Overview](#overview)
- [Critical Implementation Fixes](#critical-implementation-fixes)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Performance](#model-performance)
- [Web Application](#web-application)

## 🎯 Overview

This project trains and compares multiple machine learning algorithms to predict Formula 1 race winners with high accuracy. The pipeline automatically selects the best-performing model and deploys it through a sleek, interactive web application.

### Key Highlights

- **Data Leakage Prevention**: Uses pre-race `cumulative_points` instead of post-race points to ensure realistic results.
- **Multi-Model Comparison**: Evaluates Random Forest, XGBoost, Logistic Regression, Gradient Boosting, and more.
- **Automatic Model Selection**: Identifies the best performer using ROC-AUC and F1-Score metrics.
- **Robust Preprocessing**: Standardized feature scaling (`StandardScaler`) and automated categorical encoding.
- **Premium Web Interface**: Modern Flask-powered web app with F1-themed dark mode and dynamic circuit metadata.
- **Real-Time Predictions**: Instant winning probability for any driver/circuit/team combination.

## ⚠️ Critical Implementation Fixes

The model includes several critical improvements over traditional F1 predictors:

1. **Fixing Data Leakage**: Replaced post-race `points` with `cumulative_points` (points earned *before* the current race). This ensures the model only uses information that would be available prior to the race start.
2. **Feature Scaling**: Implemented `StandardScaler` to normalize features like `cumulative_points` (0-400+) and `grid_position` (1-20), preventing large-scale variables from dominating the model.
3. **Class Imbalance Handling**: Applied `class_weight='balanced'` and optimized for `Weighted F1-Score` to account for the fact that only one driver wins per race.

## 📊 Dataset

The dataset is sourced from the [Kaggle Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), compiled from the [Ergast API](http://ergast.com/mrd/).

The pipeline integrates 14 CSV files:

1. **circuits.csv**: Circuit metadata (location, country, coordinates).
2. **constructor_results.csv**: Constructor race points.
3. **constructor_standings.csv**: Constructor championship positions.
4. **constructors.csv**: Team names and nationalities.
5. **driver_standings.csv**: Driver championship points and wins.
6. **drivers.csv**: Driver names, nationalities, and DOB.
7. **lap_times.csv**: Lap-by-lap timing data.
8. **pit_stops.csv**: Pit stop durations and lap numbers.
9. **qualifying.csv**: Q1, Q2, and Q3 session times.
10. **races.csv**: Race calendar and metadata.
11. **results.csv**: Final race results (primary target data).
12. **seasons.csv**: Historical season links.
13. **sprint_results.csv**: Sprint race outcome data.
14. **status.csv**: Finish status codes (Finished, DNF, etc.).

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/heyisula/f1.git
   cd f1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Step 1: Train the Model

Open and run all cells in `train.ipynb`:

```bash
 jupyter notebook train.ipynb
```

The notebook will:
- Load and preprocess data from the `data/` directory.
- Implement data leakage fixes and feature engineering.
- Train multiple models (XGBoost, RandomForest, Gradient Boosting).
- Save the best model and preprocessors to `out/models/f1_model_data.pkl`.

### Step 2: Launch the Web App

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`. You can:
- Select a driver, team, and circuit.
- Watch as the **Laps** field auto-fills based on the circuit selected.
- Get an AI-predicted winning probability.

## 🔧 Features

### Current Feature Set

| Feature | Description | Importance |
|---------|-------------|------------|
| **Grid Position** | Starting position on the grid (1-20) | High |
| **Cumulative Points** | Points earned *entering* the race | High |
| **Circuit ID** | Encoded circuit identifier | Medium |
| **Driver/Team ID** | Encoded driver and constructor identifiers | Medium-High |
| **Driver Age** | Calculated age at race time | Low |
| **Laps** | Total race distance | Low |

### Potential Future Features
- Qualifying gap to pole (seconds)
- Reliability index (DNF rate over last 5 races)
- Teammate vs. Teammate historical performance
- Weather conditions (Rain/Dry probability)

## 📊 Model Performance

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **Weighted F1-Score** | ~0.96 | High overall classification accuracy |
| **ROC-AUC** | ~0.95 | Excellent model discriminative ability |
| **Winner Precision** | ~0.55-0.65 | Realistic given single-winner probability |

> Note: Metrics represent the optimized **Gradient Boosting** model.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Ergast F1 API](http://ergast.com/mrd/) via [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- Inspiration: F1 racing analytics community
