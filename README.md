# Personality Detection Using Machine Learning

## Project Overview
This project aims to analyze and predict personality types using various classification algorithms, including Decision Trees, Random Forest, and Support Vector Machines. The goal is to identify key features that can help in classifying different personality types.

## Dataset
- **Description**: The dataset used in this project consists of 5000 samples, each representing a unique individual. The dataset includes 8 features, such as demographic information, response patterns, and interaction style indicators.
- **Source**: The dataset was obtained from a publicly available repository.
- **Size and Structure**: The dataset has 128061 rows and 9 columns.
- **Features Included**: The features included in the dataset are:
  - Demographic information (age, sex, education level)
  - Personality Scores: The features included Introversion,Sensing,Thinking,Judging
  - Interest Category: Type of Interest (eg.,Sports,Arts,Technology)
  - Presonality Type: Personality category to be predicted (eg.,ENFP,INFP)
## Features
- **Key Features**: The key features used in the model are:
  - Demographic information
  - Response patterns
  - Interaction style indicators
- **Feature Engineering**: The following feature engineering techniques were used:
  - Data normalization
  - Feature scaling
  - Encoding categorical variables
  - Age Group
- **Data Preprocessing**: The following data preprocessing steps were taken:
  - Handling missing values
  - Encoding categorical variables

## Models
Implementation of three different classifiers:
- **Decision Tree Classifier**
  - Accuracy: 86%
- **Random Forest Classifier**
  - Accuracy: 90%
- **Support Vector Machine (SVM)**
  - Accuracy: 87%

## Results
- **Model Performance**: The Random Forest Classifier performed the best, with an accuracy of 86%.
- **Classification Reports**: The classification reports for each model are:
  - Decision Tree Classifier: 86% accuracy, 79% precision, 78% recall, 78% F1-score
  - Random Forest Classifier: 90% accuracy, 83% precision, 86% recall, 85% F1-score
  - Support Vector Machine (SVM): 86% accuracy, 80% precision, 77% recall, 79% F1-score
- **Visualizations**: The visualizations of the results are:
  - Confusion matrices for each model
  - ROC curves for each model

