# Credit Card Fraud Detection using Machine Learning & Deep Learning

_________________________________________________________________________

This project implements a comprehensive approach to detecting credit card fraud by leveraging both traditional machine learning algorithms and deep learning models. The primary objective is to accurately identify fraudulent transactions from a highly imbalanced dataset, thereby reducing financial losses and improving security for financial institutions.

## Overview

Fraud detection is a critical component in financial security. This project:
- Processes and explores the Kaggle credit card fraud dataset.
- Applies advanced data preprocessing techniques including feature scaling and undersampling to address class imbalance.
- Implements multiple classifiers ranging from baseline models (e.g., Logistic Regression, SVM) to ensemble methods (Random Forest, XGBoost) and neural network approaches.
- Compares model performance using key evaluation metrics.
- Develops a deep neural network using TensorFlow/Keras that is tailored to capture complex, non-linear relationships in the data.

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features:**
  - **Time:** Seconds elapsed between each transaction and the first transaction in the dataset.
  - **V1-V28:** Principal components obtained via PCA (to protect sensitive information).
  - **Amount:** Transaction amount.
  - **Class:** Binary label (0: legitimate, 1: fraudulent).
- **Data Processing:**
  - **Dropping Non-informative Features:** The 'Time' column is removed.
  - **Feature Scaling:** The 'Amount' feature is standardized using `StandardScaler` and then the original 'Amount' is dropped.
  - **Undersampling:** To mitigate the issue of severe class imbalance, the majority class is randomly undersampled using `RandomUnderSampler` from the `imblearn` package.

## Methodology

### Data Preprocessing
- **Standardization:**  
  The 'Amount' feature is scaled to have zero mean and unit variance.
- **Undersampling:**  
  Random undersampling is applied to balance the classes, which is crucial for training robust classifiers.

### Model Implementation

The project evaluates several models to benchmark performance:

1. **Logistic Regression:**  
   A baseline model to set a performance standard.
   
2. **Support Vector Machine (SVM):**  
   Implements probability estimates for fraud classification.
   
3. **Ensemble Learning - Bagging (Random Forest):**  
   Aggregates multiple decision trees to improve prediction stability.
   
4. **Ensemble Learning - Boosting (XGBoost):**  
   Uses gradient boosting to enhance accuracy.
   
5. **Multi-Layer Perceptron (MLP):**  
   A neural network model implemented with scikit-learn that serves as another baseline.
   
6. **Deep Neural Network (TensorFlow/Keras):**  
   **Architecture Details:**
   - **Input Layer:** Accepts 29 features.
   - **Hidden Layers:** 
     - Dense layer with 32 neurons and `relu` activation, followed by a Dropout layer (20% dropout).
     - Dense layer with 16 neurons (`relu`), followed by Dropout.
     - Dense layer with 8 neurons (`relu`), followed by Dropout.
     - Dense layer with 4 neurons (`relu`), followed by Dropout.
   - **Output Layer:** A single neuron with `sigmoid` activation for binary classification.
   - **Training Parameters:**
     - Optimizer: Adam (learning rate = 0.001)
     - Loss Function: Binary Crossentropy
     - Early Stopping: Monitors validation accuracy with a patience of 15 epochs.
     - Training: 6 epochs, batch size of 5, with a 15% validation split.

### Evaluation Metrics

For each model, the following metrics are computed to evaluate performance:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**
- **Confusion Matrix**
- **ROC and Precision-Recall Curves**

## Technical Findings

- **Baseline Performance:**  
  Logistic Regression and SVM provide solid initial benchmarks, while ensemble methods like Random Forest and XGBoost further improve detection capabilities.
  
- **Neural Network Insights:**  
  The deep learning model (TensorFlow/Keras) demonstrates competitive performance with high precision, recall, and AUC. Its multi-layer architecture, combined with dropout regularization, effectively captures complex patterns within the imbalanced dataset.
  
- **Imbalanced Data Handling:**  
  Random undersampling significantly improves the models’ sensitivity to the minority (fraud) class, ensuring that the classifiers do not simply default to the majority class.

## Business Insights

- **Fraud Prevention:**  
  Early detection of fraudulent transactions can greatly minimize financial losses.
- **Operational Efficiency:**  
  Automating fraud detection reduces the need for manual review and accelerates response times.
- **Strategic Decision-Making:**  
  Insights from model performance can help financial institutions adjust risk management strategies and improve overall security protocols.

## Hardware & Software Requirements

- **Hardware Requirements:**
  - Multi-core CPU
  - At least 8GB of RAM (more recommended for deep learning model training)
  - Sufficient disk space for storing datasets and model artifacts
  
- **Software Requirements:**
  - **Operating System:** Windows, macOS, or Linux
  - **Python Version:** 3.7+
  - **Key Libraries:**
    - `numpy`, `pandas` for data manipulation
    - `matplotlib`, `seaborn` for visualization
    - `scikit-learn` for classical machine learning models
    - `imbalanced-learn` for handling class imbalance
    - `xgboost` for boosting algorithms
    - `tensorflow` and `keras` for deep learning

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd CreditCardFraudDetection
