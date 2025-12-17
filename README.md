# 🏎️ Formula 1 Race Winner Prediction Model

A comprehensive machine learning project that predicts Formula 1 race winners using historical data from 1950 to 2024. The model leverages 52+ engineered features from multiple data sources including race results, qualifying times, pit stops, lap times, sprint races, and championship standings.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)

## 🎯 Overview

This project builds and compares 10 different machine learning models to predict Formula 1 race winners. The system automatically selects the best-performing model based on comprehensive evaluation metrics including accuracy, F1-score, precision, recall, and ROC-AUC.

### Key Highlights

- **Comparision of 10 Machine Learnining Models**: Random Forest, XGBoost, Gradient Boosting, Extra Trees, AdaBoost, Decision Tree, Logistic Regression, KNN, Naive Bayes, and SVM
- **52+ Engineered Features**: Comprehensive feature engineering from 12 different data sources
- **Time-Based Validation**: Proper temporal split to prevent data leakage
- **Automatic Best Model Selection**: Identifies the optimal model based on test performance
- **Real-World Application**: Can predict winners for future races given driver and race data

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

## 🔧 Features

### Feature Categories (52+ total features)

#### 1. Grid & Qualifying Features
- Starting grid position
- Qualifying position (final)
- Q1, Q2, Q3 lap times
- Best qualifying time

#### 2. Sprint Race Features
- Sprint race position
- Sprint points earned
- Sprint grid position
- Sprint race participation flag

#### 3. Lap Time & Performance Features
- Average lap time
- Lap time standard deviation (consistency)
- Fastest/slowest lap times
- Total laps completed
- Lap time consistency metric

#### 4. Pit Stop Strategy Features
- Number of pit stops
- Average pit stop duration
- Total time lost in pits
- Fastest/slowest pit stop

#### 5. Driver Historical Performance
- Previous wins, podiums, top 5, top 10 finishes
- Total career points
- Average finishing position
- Recent form (last 5 and 10 races)
- Win rate and podium rate
- DNF (Did Not Finish) rate
- Average starting grid position
- Average qualifying position

#### 6. Constructor/Team Performance
- Team's previous wins and podiums
- Team's total points
- Team's average finishing position
- Team's recent form
- Team win rate and podium rate

#### 7. Circuit-Specific Performance
- Driver's races at specific circuit
- Driver's wins at specific circuit
- Driver's podiums at specific circuit
- Driver's average position at circuit
- Circuit-specific win rate

#### 8. Championship Context
- Championship points before race
- Championship position before race
- Championship wins before race

#### 9. Encoded Categorical Features
- Driver ID (encoded)
- Constructor ID (encoded)
- Circuit ID (encoded)
