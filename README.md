# Titanic-Survival-Prediction-Random-Forest-Classifier-vs.-Logistic-Regression-


# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using scikit-learn pipelines and multiple classification algorithms.

## Project Overview

This project analyzes the famous Titanic dataset to build predictive models that determine whether a passenger survived the disaster. The project demonstrates proper machine learning workflow including data preprocessing, feature engineering, model training, and evaluation.

## Features

- **Data Preprocessing Pipeline**: Automated handling of numerical and categorical features
- **Missing Value Imputation**: Strategic handling of missing data using median/mode imputation
- **Feature Engineering**: One-hot encoding for categorical variables and standardization for numerical features
- **Model Comparison**: Implementation and comparison of Random Forest and Logistic Regression classifiers
- **Hyperparameter Tuning**: Grid search with cross-validation for optimal model performance
- **Model Evaluation**: Comprehensive evaluation using classification reports, confusion matrices, and feature importance analysis

## Dataset

The Titanic dataset includes passenger information such as:
- `survived`: Target variable (0 = No, 1 = Yes)
- `pclass`: Ticket class
- `sex`: Gender
- `age`: Age in years
- `sibsp`: Number of siblings/spouses aboard
- `parch`: Number of parents/children aboard
- `fare`: Passenger fare
- Additional features like `class`, `who`, `adult_male`, `alone`

## Models Implemented

### 1. Random Forest Classifier
- **Accuracy**: 81.34%
- Grid search optimization for `n_estimators`, `max_depth`, and `min_samples_split`
- Feature importance analysis showing `fare`, `age`, and `sex_male` as top predictors

### 2. Logistic Regression
- **Accuracy**: 82.09%
- Regularization techniques (L1/L2 penalty)
- Class weight balancing options
- Coefficient analysis for feature interpretation

## Key Results

- Both models achieved similar performance (~81-82% accuracy)
- Gender (`sex`) emerged as the strongest predictor of survival
- Passenger class and fare also showed significant predictive power
- Proper preprocessing pipeline ensured robust model performance

## Technical Implementation

### Preprocessing Pipeline
```python
# Numerical features: median imputation + standardization
# Categorical features: mode imputation + one-hot encoding
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])
```

### Model Pipeline
```python
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
seaborn
```

## Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

3. Run the Jupyter notebook:
```bash
jupyter notebook RandomForestClassifier.ipynb
```

## Project Structure

```
titanic-survival-prediction/
├── RandomForestClassifier.ipynb    # Main analysis notebook
├── README.md                       # Project documentation
└── requirements.txt               # Python dependencies
```

## Key Learnings

- **Pipeline Implementation**: Demonstrates proper use of scikit-learn pipelines for reproducible ML workflows
- **Feature Engineering**: Shows effective handling of mixed data types (numerical and categorical)
- **Model Comparison**: Illustrates systematic approach to comparing different algorithms
- **Cross-Validation**: Implements stratified k-fold CV for robust model evaluation
- **Hyperparameter Tuning**: Uses GridSearchCV for optimal model configuration

## Future Improvements

- [ ] Feature selection techniques to identify optimal feature subset
- [ ] Additional algorithms (SVM, XGBoost, Neural Networks)
- [ ] Ensemble methods combining multiple models
- [ ] Advanced feature engineering (interaction terms, polynomial features)
- [ ] Model deployment using Flask/FastAPI

## Contributing

Feel free to fork this project and submit pull requests for any improvements.

## License

This project is open source and available under the [MIT License](LICENSE).

---

*This project was created as part of learning machine learning fundamentals and scikit-learn implementation.*
