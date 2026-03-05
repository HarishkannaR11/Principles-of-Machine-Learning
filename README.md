# Machine Learning Laboratory Experiments

This repository contains various machine learning experiments demonstrating different algorithms and techniques for classification and regression tasks.

## Repository Structure

```
Principles-of-Machine-Learning/
├── Exp1/           # Initial Classification Experiments
├── Exp2/           # Loan Prediction (Regression)
├── Exp3/           # Email Spam Classification
├── Exp4/           # Cancer Detection
├── Exp5/           # Handwritten Digit Recognition
├── Exp6/           # Principal Component Analysis (PCA)
└── README.md
```

---

## Experiment 1: Classification

### Datasets
- **Diabetes**: Classification of diabetes cases
- **Email Spam**: Spam detection
- **Handwritten Digits**: MNIST digit recognition
- **Iris**: Flower species classification
- **Loan**: Loan approval prediction

### Key Features
- Exploratory Data Analysis (EDA)
- Data visualization
- Basic classification models
- Model evaluation

---

## Experiment 2: Loan Prediction (Regression)

**Objective**: Predict loan amounts and approval using various machine learning models.

### Dataset
- Training data: `train.csv`
- Test data: `test.csv`

### Models Implemented
1. **Linear Regression**: For continuous loan amount prediction
2. **Logistic Regression**: For loan approval classification
3. **Decision Tree Classifier**: Tree-based classification approach
4. **Random Forest Classifier**: Ensemble of decision trees
5. **K-Nearest Neighbors (KNN)**: Instance-based learning
6. **Support Vector Classifier (SVC)**: Kernel-based classification

### Preprocessing Techniques
- **Handling Missing Data**: 
  - `fillna()` for numeric features
  - Mode imputation for categorical features
  - `dropna()` for critical columns
- **Feature Encoding**: 
  - OneHotEncoder for categorical variables
  - `get_dummies()` for creating dummy variables
- **Feature Scaling**: 
  - StandardScaler (z-score normalization)
  - MinMaxScaler (min-max normalization)
- **Data Splitting**: train_test_split with validation sets

### Key Files
- `Loanregression.ipynb`: Main regression notebook
- `looan.ipynb`: Alternative loan analysis
- `Loan.ipynb`: Additional loan experiments
- `loan_predictions.csv`: Model predictions output

---

## Experiment 3: Email Spam Classification

**Objective**: Classify emails as spam or not spam using various classification algorithms.

### Dataset
- `spambase_csv.csv`: Email feature dataset

### Models Implemented

#### 1. Naive Bayes Classifiers
- **GaussianNB**: Assumes Gaussian distribution
- **MultinomialNB**: For discrete count features
- **BernoulliNB**: For binary/boolean features

#### 2. K-Nearest Neighbors (KNN)
Multiple algorithms tested:
- **Brute Force**: Exhaustive search
- **KD-Tree**: Efficient for low dimensions
- **Ball Tree**: Better for higher dimensions

#### 3. Support Vector Machines (SVM)
Multiple kernel functions:
- **Linear**: For linearly separable data
- **Polynomial (poly)**: Degree=3, for non-linear patterns
- **RBF (Radial Basis Function)**: Most versatile kernel
- **Sigmoid**: Neural network-inspired kernel

### Preprocessing Techniques
- **Feature Scaling**: 
  - StandardScaler for normalization
  - MinMaxScaler for [0,1] range scaling
- **Data Splitting**: train_test_split for train/test sets

### Model Evaluation
Each model is evaluated using:
- Accuracy metrics
- Classification reports
- Confusion matrices
- Comparative performance analysis

### Key Files
- `email_classf.ipynb`: Main classification notebook
- `spambase_csv.csv`: Feature dataset

---

## Experiment 4: Cancer Detection

**Objective**: Breast cancer classification using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

### Dataset
- `wdbc.csv`: Breast cancer diagnostic data

### Features
- Medical diagnostic measurements
- Binary classification (Malignant/Benign)

---

## Experiment 5: Handwritten Digit Recognition

**Objective**: Recognize handwritten digits using computer vision techniques.

### Dataset
- English handwritten character dataset

### Approach
- Image preprocessing
- Feature extraction
- Classification models

### Key Files
- `EnglishHandwritten.ipynb`: Main recognition notebook

---

## Experiment 6: Principal Component Analysis (PCA)

**Objective**: Dimensionality reduction and feature extraction using PCA.

### Dataset
- `wdbc.csv`: Using cancer dataset for demonstration

### Techniques
- Variance analysis
- Feature reduction
- Visualization in reduced dimensions

### Key Files
- `PCA.ipynb`: PCA implementation and analysis

---

## Common Dependencies

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.decomposition import PCA

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## Usage

Each experiment is contained in its respective folder with Jupyter notebooks. To run:

```bash
jupyter notebook
# Navigate to the desired experiment folder and open the .ipynb file
```

---

## Results and Outputs

- **HTML Reports**: Each experiment includes HTML exports of the notebooks
- **CSV Outputs**: Model predictions saved as CSV files
- **Visualizations**: Graphs and plots for data analysis
- **Documentation**: PDF files in the `pdf/` folder

---

## Model Performance Summary

### Experiment 2 (Loan Prediction)
- Multiple models compared for regression and classification tasks
- Feature engineering with OneHotEncoder improves model performance
- Scaling techniques critical for distance-based algorithms (KNN, SVC)

### Experiment 3 (Email Spam)
- Naive Bayes variants perform well on text-based features
- SVM with RBF kernel shows strong performance
- KNN with appropriate algorithm choice (kd_tree/ball_tree) efficient

---

## Key Learning Outcomes

1. **Data Preprocessing**: 
   - Handling missing data
   - Feature encoding
   - Feature scaling

2. **Model Selection**:
   - Understanding different algorithm types
   - Choosing appropriate models for tasks
   - Hyperparameter tuning

3. **Evaluation**:
   - Cross-validation techniques
   - Performance metrics
   - Model comparison

4. **Advanced Techniques**:
   - Ensemble methods (Random Forest)
   - Dimensionality reduction (PCA)
   - Kernel methods (SVM)

---

## Contributing

This repository is part of academic laboratory work. For improvements or suggestions, please follow academic collaboration guidelines.

---

## License

Academic use only - Part of ML Laboratory coursework.

---

## Acknowledgments

- Dataset sources: UCI Machine Learning Repository, Kaggle
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn