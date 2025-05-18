# 🦟 Dengue Prediction using Machine Learning

This project aims to forecast the number of dengue fever cases in San Juan and Iquitos based on environmental and climatic conditions. Built as part of the [DrivenData DengAI Challenge](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/), this model can assist public health authorities in taking early action to prevent outbreaks.


## 📂 Dataset

- **Source**: DrivenData DengAI competition
- **Files Used**:
  - `dengue_features_train.csv`: Climate and environmental features for training.
  - `dengue_labels_train.csv`: Weekly dengue case counts.
  - `dengue_features_test.csv`: Features for test prediction.
  - `submission.csv`: Final prediction output format.


## 📈 Problem Statement

Given historical weather, climate, and disease data, the task is to predict the weekly number of dengue cases for two cities (San Juan and Iquitos) to help guide public health interventions.


## ⚙️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (for EDA and visualization)


## 🧠 ML Approach

1. **Data Cleaning**
   - Handled missing values with forward/backward filling.
   - Converted date features and removed irrelevant columns.

2. **Feature Engineering**
   - Normalization of climate variables.
   - City-based data splitting for city-specific modeling.

3. **Modeling**
   - Applied Random Forest Regressor for its robustness.
   - Performed cross-validation for evaluation.

4. **Prediction**
   - Generated weekly dengue case predictions.
   - Saved results in the required `submission.csv` format.
