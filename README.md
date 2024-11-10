# Personality Detection Using Machine Learning

## Project Overview
This project focuses on building a machine learning model to predict personality types based on user responses and demographic data. Various classification algorithms were explored, including Decision Trees, Random Forest, and Support Vector Machines (SVM). The objective was to identify key features that contribute to accurate personality classification and evaluate the performance of different models.

## Dataset
### Dataset Description
- **Source**: Publicly available repository.
- **Size and Structure**: The dataset consists of **128,061 rows** and **9 columns**, encompassing data from **5,000 unique individuals**.
- **Features**:
  - **Demographic Information**: Age, Gender, Education Level
  - **Personality Scores**: Scores on traits like Introversion, Sensing, Thinking, and Judging
  - **Interest Category**: Type of interest (e.g., Sports, Arts, Technology)
  - **Personality Type**: The target variable for prediction (e.g., ENFP, INFP)

## Features
### Key Features
- **Demographic Information**: Includes user age, sex, and education level.
- **Response Patterns**: Analyzes individual response tendencies across the dataset.
- **Interaction Style Indicators**: Evaluates user interaction style based on their input.

### Feature Engineering
The following techniques were applied to improve model performance:
- **Data Normalization**: Ensured consistency in numerical features.
- **Feature Scaling**: Standardized features for better model convergence.
- **Encoding Categorical Variables**: Used One-Hot Encoding for categorical data.
- **Age Group Binning**: Created age groups for better demographic analysis.

### Data Preprocessing
Key preprocessing steps taken include:
- **Handling Missing Values**: Used mean/mode imputation for missing entries.
- **Encoding Categorical Variables**: Applied One-Hot Encoding for categorical features.
- **Data Splitting**: Split the data into training (80%) and testing (20%) sets.

## Models
Three different classification models were implemented:
1. **Decision Tree Classifier**
   - **Accuracy**: 86%
   - **Precision**: 79%
   - **Recall**: 78%
   - **F1-Score**: 78%

2. **Random Forest Classifier**
   - **Accuracy**: 90%
   - **Precision**: 83%
   - **Recall**: 86%
   - **F1-Score**: 85%

3. **Support Vector Machine (SVM)**
   - **Accuracy**: 87%
   - **Precision**: 80%
   - **Recall**: 77%
   - **F1-Score**: 79%

## Results
### Model Performance
The **Random Forest Classifier** outperformed the other models, achieving the highest accuracy of **90%**. The performance of each model is summarized below:
- **Decision Tree Classifier**: 86% accuracy, suitable for basic classification tasks.
- **Random Forest Classifier**: 90% accuracy, best performance due to ensemble learning.
- **Support Vector Machine (SVM)**: 87% accuracy, effective for high-dimensional data.

### Classification Reports
Detailed performance metrics for each model:
- **Decision Tree**: 79% precision, 78% recall, 78% F1-score
- **Random Forest**: 83% precision, 86% recall, 85% F1-score
- **SVM**: 80% precision, 77% recall, 79% F1-score

### Visualizations
- **Confusion Matrices**: Provided for each model to illustrate classification performance.
- **ROC Curves**: Plotted to compare model performance based on True Positive and False Positive rates.

## Conclusion
The **Random Forest Classifier** achieved the highest accuracy and F1-score, making it the most suitable model for personality prediction in this dataset. Future improvements could include hyperparameter tuning and incorporating additional features for enhanced performance.

## Future Work
- **Hyperparameter Tuning**: Optimize model parameters for improved accuracy.
- **Additional Features**: Explore new features like user behavioral data for better predictions.
- **Deployment**: Consider deploying the model using a web application for real-time personality analysis.

