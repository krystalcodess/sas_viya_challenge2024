# SAS Viya for Learners Challenge 2024 - Data Analysis and Modeling

This Jupyter Notebook contains the code for a machine learning solution developed for the SAS Viya for Learners Challenge 2024. The goal was to predict loan default risk (`default`) based on a dataset containing financial and credit-related features. Below is a summary of the key steps performed:

## 1. Key Libraries Used
- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical operations.
- `matplotlib` and `seaborn`: Data visualization.
- `sklearn`: Machine learning (preprocessing, model training, evaluation, and hyperparameter tuning).

## 2. Approach

### 2.1 Data Loading and Exploration
- **Dataset**: Loaded the training dataset.
- **Exploration**:
  - Displayed the first few rows of the dataset to understand its structure.
  - Checked the distribution of key variables like `late_payments` and `derogatory_reports` using `value_counts()`.
  - Identified missing values in the dataset using `df.isnull().sum()`.
  - Reviewed dataset schema and data types with `df.info()`.

### 2.2 Visualization
- **Exploratory Visualization**: Generated a boxplot to visualize the relationship between variables using `seaborn` and `matplotlib`.

### 2.3 Data Preprocessing
- **Handling Missing Values**:
  - Defined a custom function `setOccupation` to impute missing values in the `occupation` column based on conditions involving `loan_reason` and `loan_amount`.
  - Created a function `Handle_missing` to preprocess the dataset, including applying `setOccupation` and handling missing values.
  - Imputed missing values in numerical columns (`mortgage_amount`, `property_value`, `derogatory_reports`, `late_payments`, `oldest_credit_line`, `recent_credit`, `credit_number`, `ratio`) with their respective medians.
- **Encoding Categorical Variables**:
  - Used `LabelEncoder` to transform the `loan_reason` column into numerical values.
  - Applied one-hot encoding to the `occupation` column using `OneHotEncoder` and concatenated the resulting dummy variables to the dataset.
- **Feature Scaling**:
  - Standardized numerical features (`loan_amount`, `mortgage_amount`, `property_value`, `oldest_credit_line`, `ratio`, `occupation_length`, `derogatory_reports`) using `StandardScaler` to ensure consistent scaling for model training.

### 2.4 Model Training
- **Train-Test Split**: Split the preprocessed training data into training and validation sets using `train_test_split` from `sklearn`.
- **Model Selection**: Used a Support Vector Classifier (`SVC`) from `sklearn.svm` for binary classification of the `default` target variable.
- **Hyperparameter Tuning**:
  - Performed grid search with `GridSearchCV` to optimize SVC hyperparameters (e.g., `C`, `kernel`, `gamma`).
  - Evaluated model performance using cross-validation (`cross_val_score`) and metrics like accuracy and classification report.
- **Model Evaluation**: Assessed the trained model's performance on the validation set.

### 2.5 Test Data Processing and Prediction
- **Test Data Loading**: Loaded the test dataset and applied the same preprocessing steps as the training data (missing value imputation, encoding, and scaling).
- **Prediction**: Used the trained SVC model to predict the `default` values for the test dataset.

## 3. Results
- Achieved an accuracy of 95.7%, securing 3rd place in the national SAS Viya for Learners Challenge 2024.
![image](https://github.com/user-attachments/assets/8fcffb26-500b-4494-9ee8-11a3e09a683d)


This notebook demonstrates a complete machine learning pipeline, from data exploration and visualization to preprocessing, model training, and prediction for the SAS Viya for Learners Challenge 2024.
