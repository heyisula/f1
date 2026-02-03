# 🏎️ Formula 1 Race Winner Prediction Model

A comprehensive machine learning project that predicts Formula 1 race winners using historical data from 1950 to 2024. The system compares multiple ML algorithms, automatically selects the best performer, and deploys it via a premium web application with a modern dark-mode interface.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Performance](#model-performance)
- [Web Application](#web-application)

## 🎯 Overview

This project trains and compares multiple machine learning algorithms to predict Formula 1 race winners with high accuracy. The pipeline automatically selects the best-performing model and deploys it through a sleek, interactive web application.

### Key Highlights

- **Multi-Model Comparison**: Trains and evaluates Random Forest, XGBoost, Logistic Regression, Decision Tree, and Naive Bayes
- **Automatic Model Selection**: Chooses the best model based on ROC-AUC score
- **Hyperparameter Optimization**: Fine-tunes the winning model using GridSearchCV
- **7 Engineered Features**: Grid position, points, laps, driver/team/circuit encoding, driver age
- **Time-Based Validation**: Temporal split (pre-2023 for training, 2023+ for testing) to prevent data leakage
- **Premium Web Interface**: Modern Flask-powered web app with F1-themed dark mode design
- **Real-Time Predictions**: Get instant winning probability for any driver/circuit/team combination

## 📊 Dataset

The dataset is sourced from [Kaggle - Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), compiled from [Ergast API](http://ergast.com/mrd/).

### Data Sources Used

The model integrates 12 CSV files:

1. **circuits.csv** - Circuit information (location, country, coordinates)
2. **constructor_results.csv** - Constructor race results
3. **constructor_standings.csv** - Constructor championship standings
4. **constructors.csv** - Constructor/team details
5. **driver_standings.csv** - Driver championship standings
6. **drivers.csv** - Driver information (nationality, DOB)
7. **lap_times.csv** - Lap-by-lap timing data
8. **pit_stops.csv** - Pit stop strategy and duration
9. **qualifying.csv** - Qualifying session times (Q1, Q2, Q3)
10. **races.csv** - Race schedule and metadata
11. **results.csv** - Final race results and positions
12. **sprint_results.csv** - Sprint race results
13. **status.csv** - Race finish status (completed, DNF, etc.)

## � Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
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
- Load and preprocess data from the `data/` directory
- Generate EDA visualizations in `out/images/`
- Train multiple ML models (Random Forest, XGBoost, etc.)
- Compare performance metrics (ROC-AUC, F1-Score, Precision, Recall)
- Select the best model and optimize its hyperparameters
- Save the final model to `models/best_f1_model.pkl`

### Step 2: Launch the Web App

Start the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

You'll see a modern, dark-mode F1 prediction interface where you can:
- Select a driver, constructor (team), and circuit
- Set grid position, laps, and current championship points
- Click **"Analyze Race"** to get the AI-predicted winning probability

## 🔧 Features

### Current Feature Set (7 Features)

The model uses the following engineered features:

1. **Grid Position** - Starting position on the grid (1-20)
2. **Championship Points** - Driver's current points in the season
3. **Laps Completed** - Number of laps in the race
4. **Driver ID (Encoded)** - Label-encoded driver identifier
5. **Constructor ID (Encoded)** - Label-encoded team identifier
6. **Circuit ID (Encoded)** - Label-encoded circuit identifier
7. **Driver Age** - Age of the driver at race time

> **Note**: These features are then standardized using `StandardScaler` for models that require normalized inputs (e.g., Logistic Regression, SVM).

### Potential Future Features

The dataset supports additional feature engineering opportunities:
- Qualifying times (Q1, Q2, Q3)
- Sprint race results
- Lap time statistics (average, consistency)
- Pit stop data (count, duration)
- Historical performance (driver/team win rates, DNF rates)
- Circuit-specific performance metrics

## 📊 Model Performance

The system compares the following models:

| Model | Approach |
|-------|----------|
| **Random Forest** | Ensemble of decision trees with class balancing |
| **XGBoost** | Gradient boosting with imbalanced class handling |
| **Logistic Regression** | Linear model with L2 regularization |
| **Decision Tree** | Single tree classifier |
| **Naive Bayes** | Probabilistic Gaussian classifier |

Evaluation metrics:
- **ROC-AUC** (primary metric for model selection)
- **F1-Score** (harmonic mean of precision and recall)
- **Precision** (accuracy of positive predictions)
- **Recall** (coverage of actual winners)

> The best model is automatically selected and optimized via GridSearchCV.

## 🌐 Web Application

### Features
- **Modern UI**: Dark-mode design with F1 red accents
- **Responsive Layout**: Works on desktop and mobile
- **Real-time Predictions**: Instant win probability calculation
- **Interactive Controls**: Sliders, dropdowns, and input fields
- **Visual Feedback**: Animated probability ring display

### Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML**: scikit-learn, XGBoost
- **Data Processing**: pandas, NumPy

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more sophisticated feature engineering
- Experiment with deep learning models
- Improve the web interface
- Add unit tests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Ergast F1 API](http://ergast.com/mrd/) via [Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- Inspiration: F1 racing analytics community
