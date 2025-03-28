## MACHINE LEARNING + Fantasy Premier League

#### Can machine learning models be used to predict FPL points of players?

This project explores the application of machine learning to predict Fantasy Premier League (FPL) player performance. The system uses historical data and advanced feature engineering to generate weekly predictions for player performance.

### Data Sources
The project utilizes two main data sources:
1. Historical FPL data from 2016/17 onwards
2. Current season data from the official FPL API

Data from the 2022/23, 2021/2022 and 2020/21 seasons were used to train the model, while the FPL API provides real-time data for the current season.

### Project Structure

#### Data Processing Pipeline

* **merge_previous_seasons_data.py**
Merges player data across multiple seasons, adding new features including:
- Total stats (points, bonus points, goals, saves)
- Team position rankings
- Player value metrics
- Historical performance indicators

* **clean_previous_seasons.ipynb**
Performs data cleaning and feature engineering:
- Removes unused columns
- Creates historical feature sets
- Calculates statistical measures
- Generates team performance metrics

#### Weekly Updates

* **weekly_fixtures.ipynb**
Retrieves upcoming gameweek data from the FPL API, including:
- Player information
- Team details
- Match schedules
- Player costs

* **weekly_results.ipynb**
Collects completed gameweek results and player performance data

* **clean_fixtures.ipynb**
Processes and prepares upcoming gameweek data for prediction

#### Modeling

* **train_model.ipynb**
Implements a dual-model approach:
1. Classification model for predicting player starts
2. Regression model for predicting player points

The system uses position-specific models for:
- Goalkeepers
- Defenders
- Midfielders
- Forwards

### Visualization
The project includes visualization tools to display:
- Top performers by position
- Feature importance analysis
- Performance trends

### Project Organization

* **datasets/**
Contains raw data from FPL API and historical sources

* **cleaned_dataset/**
Stores processed and feature-engineered data

* **predicted_dataset/**
Holds model predictions for each gameweek

* **plots/**
Contains generated visualizations and charts

### Technical Features
- Advanced feature engineering
- Position-specific modeling
- Automated data pipeline
- Weekly prediction updates
- Performance visualization

