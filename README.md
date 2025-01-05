# Fetal Health Classification

This project aims to classify fetal health status using various machine learning models.The dataset contains several features derived from cardiotocograms, and the target variable is the fetal health status, which is categorized as 1 (Normal), 2 (Suspect), and 3 (Pathological). 

## Dataset

The dataset used is `fetal_health.csv`, which includes 2126 records with 21 features related to fetal health derived from cardiotocograms. The target variable, `fetal_health`, indicates the health status of the fetus.

## Features
- `baseline_value`: FHR baseline (beats per minute)
- `accelerations`: Number of accelerations per second
- `fetal_movement`: Number of fetal movements per second
- `uterine_contractions`: Number of uterine contractions per second
- `light_decelerations`: Number of light decelerations per second
- `severe_decelerations`: Number of severe decelerations per second
- `prolongued_decelerations`: Number of prolonged decelerations per second
- `abnormal_short_term_variability`: Percentage of time with abnormal short-term variability
- `mean_value_of_short_term_variability`: Mean value of short-term variability
- `percentage_of_time_with_abnormal_long_term_variability`: Percentage of time with abnormal long-term variability
- `mean_value_of_long_term_variability`: Mean value of long-term variability
- `histogram_width`: Width of FHR histogram
- `histogram_min`: Minimum (low frequency) of FHR histogram
- `histogram_max`: Maximum (high frequency) of FHR histogram
- `histogram_number_of_peaks`: Number of histogram peaks
- `histogram_number_of_zeroes`: Number of histogram zeros
- `histogram_mode`: Histogram mode
- `histogram_mean`: Histogram mean
- `histogram_median`: Histogram median
- `histogram_variance`: Histogram variance
- `histogram_tendency`: Histogram tendency
- `fetal_health`: Target variable indicating fetal health status (1: Normal, 2: Suspect, 3: Pathological)

## Data Preprocessing

- **Data Cleaning**: Checked for missing values and duplicated rows.
- **Standardization**: Standardized the features using `StandardScaler` to ensure that all features contribute equally to the analysis.

## Exploratory Data Analysis (EDA)

- **Distribution Plots**: Used seaborn and matplotlib to visualize the distributions of various features.
- **Correlation Analysis**: Created a correlation matrix and heatmap to identify relationships between features and the target variable.
- **Feature Plots**: Visualized the relationship between selected features and the target variable using scatter plots and joint plots.

## Machine Learning Models

Several machine learning models were implemented and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)

## Model Evaluation

- **Cross-Validation**: Evaluated models using 10-fold cross-validation on the training data.
- **Accuracy**: Compared the accuracy of different models.
- **KNN Performance**: KNN with k=7 showed the best performance with an accuracy of approximately 83%.

## Results

- **KNN Model**: The KNN model with k=7 achieved an accuracy of 83% on the test data.
- **Feature Importance**: Identified key features such as `prolongued_decelerations` and `abnormal_short_term_variability` that have a higher correlation with fetal health status.

## Visualizations

- **Boxen Plots**: Visualized the distribution of standardized features to detect outliers.
- **Count Plot**: Visualized the distribution of the target variable.
- **Correlation Heatmap**: Showed the correlation between features.
- **Scatter Plots**: Illustrated relationships between important features and the target variable.

## Conclusion

This project successfully implemented and evaluated multiple machine learning models to classify fetal health status. The KNN model with k=7 provided the highest accuracy. Further improvements can be made by exploring more advanced models and feature engineering techniques.

## Usage

1. **Data Preprocessing**: Ensure the data is cleaned and standardized.
2. **Model Training**: Use the provided pipelines to train various models.
3. **Model Evaluation**: Evaluate models using cross-validation and accuracy metrics.
4. **Visualization**: Use the provided visualization code to analyze feature distributions and relationships.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
```
