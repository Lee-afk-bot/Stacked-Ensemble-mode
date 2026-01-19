# Stacked Ensemble Model for Mullite Whisker Morphology Prediction

This repository contains the source code for the paper "Synthesis parameters and morphological dimensions of mullite whiskers".

## Description
The model uses a heterogeneous stacked ensemble approach (Level-0: XGBoost, LightGBM, CatBoost, RF, MLP, DT; Level-1: XGBoost) to predict the length, width, and aspect ratio of mullite whiskers based on synthesis parameters.

## Requirements
- Python 3.8+
- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- catboost

## Usage
Run the script directly:
python stacked_ensemble_model.py
