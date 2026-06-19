# 🏦 Loan Status Prediction

A machine learning project that predicts whether a loan application will be **approved (Y)** or **rejected (N)** based on applicant information such as income, credit history, marital status, education, and property area. The model is built using a **Support Vector Machine (SVM)** classifier with a linear kernel.

## 📋 Overview

This notebook follows an end-to-end machine learning workflow:

1. Import libraries
2. Load and inspect the dataset
3. Exploratory Data Analysis (EDA)
4. Data preprocessing and feature engineering
5. Train-test split
6. Model training
7. Model evaluation
8. Prediction

## 📊 Dataset

The project uses the **Loan Prediction Dataset** (`Loan_Predication_Dataset.csv`), which contains 614 rows and 13 columns:

| Column | Description |
|---|---|
| `Loan_ID` | Unique loan application ID |
| `Gender` | Applicant gender (Male/Female) |
| `Married` | Marital status (Yes/No) |
| `Dependents` | Number of dependents (0, 1, 2, 3+) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Yes/No |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Loan amount requested |
| `Loan_Amount_Term` | Term of the loan (in months) |
| `Credit_History` | Whether the applicant has a credit history (1.0/0.0) |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | Target variable — loan approved (Y) or not (N) |

> **Note:** The dataset file is not included in this repository. Place `Loan_Predication_Dataset.csv` in the same directory as the notebook before running it.

## 🛠️ Tech Stack

- **Python 3**
- **pandas** & **NumPy** — data manipulation
- **seaborn** & **matplotlib** — data visualization
- **scikit-learn** — model training and evaluation (`train_test_split`, `svm.SVC`, `accuracy_score`)

## 🔍 Workflow Details

### 1. Data Cleaning
- Missing values are dropped using `dropna()`, reducing the dataset from 614 to 480 rows.
- The `Loan_Status` target column is encoded: `N → 0`, `Y → 1`.
- The `"3+"` category in `Dependents` is replaced with `4` to make the column numeric.

### 2. Exploratory Data Analysis
- Count plots are used to visualize the relationship between `Loan_Status` and:
  - `Education`
  - `Married` status

### 3. Feature Encoding
Categorical features are mapped to numeric values:

| Feature | Mapping |
|---|---|
| `Married` | No → 0, Yes → 1 |
| `Gender` | Female → 0, Male → 1 |
| `Self_Employed` | No → 0, Yes → 1 |
| `Property_Area` | Rural → 0, Semiurban → 1, Urban → 2 |
| `Education` | Not Graduate → 0, Graduate → 1 |

### 4. Train-Test Split
- `Loan_ID` and `Loan_Status` are dropped from the feature set (`X`); `Loan_Status` becomes the target (`y`).
- Data is split 90/10 (`test_size=0.1`) using stratified sampling on the target, with `random_state=2` for reproducibility.

### 5. Model Training
A **Support Vector Classifier** with a **linear kernel** is trained on the training data:

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
```

## 📈 Results

| Dataset | Accuracy |
|---|---|
| Training set | **79.86%** |
| Test set | **83.33%** |

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas seaborn matplotlib scikit-learn jupyter
```

### Running the Notebook

1. Clone this repository
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Place `Loan_Predication_Dataset.csv` in the project directory.
3. Launch Jupyter and run the notebook:
   ```bash
   jupyter notebook Loan_status_prediction.ipynb
   ```

## 📁 Project Structure

```
.
├── Loan_status_prediction.ipynb   # Main notebook with full workflow
├── Loan_Predication_Dataset.csv   # Dataset (not included, add manually)
└── README.md                      # Project documentation
```

## 🔮 Possible Improvements

- Use imputation (mean/median/mode) instead of dropping missing rows to retain more data.
- Try other classifiers (Logistic Regression, Random Forest, Gradient Boosting) and compare performance.
- Apply feature scaling, which can improve SVM performance.
- Perform hyperparameter tuning (e.g., `GridSearchCV`) on the SVM kernel and `C` parameter.
- Add a confusion matrix and classification report (precision, recall, F1-score) for deeper evaluation.
