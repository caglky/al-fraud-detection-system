# AI-Based Credit Card Fraud Detection System

This project implements a robust end-to-end pipeline to detect fraudulent transactions using an Artificial Neural Network (ANN).

## Problem 
Credit card fraud is a significant challenge for financial institutions. The main technical hurdle is the extreme class imbalance, where fraudulent transactions represent a tiny fraction (often <1%) of the total data. This project aims to build a model that can accurately distinguish between legitimate and fraudulent activities by learning hidden patterns in transaction data.

## Dataset
The project utilizes a dataset containing transactions made by European cardholders.

Features: It includes 28 PCA-transformed variables (V1 to V28), 'Time', and 'Amount'.

Target: The 'Class' column (1 for Fraud, 0 for Normal).

Preprocessing: The data is normalized using StandardScaler to ensure the neural network treats all features with equal importance regardless of their original scale.

##  Features
- **Exploratory Data Analysis (EDA):** Visualization of class distribution, transaction amounts, and time-based patterns.
- **Data Preprocessing:** Automated scaling using `StandardScaler` and robust data splitting (70% Train, 15% Validation, 15% Test).
- **Deep Learning Model:** A multi-layer ANN architecture with Dropout layers to prevent overfitting.
- **Interactive Visualization:** Training history (Loss/Accuracy) curves and Confusion Matrix.
- **Risk Scoring:** Real-time risk estimation and reporting for individual transactions.

## Model 

The system uses a Sequential Neural Network architecture:

Input Layer: Matches the feature count of the dataset.

Hidden Layers: Three Dense layers with ReLU activation (32, 16, and 8 neurons).

Regularization: Dropout layers (20%) are integrated after each hidden layer to prevent overfitting and improve generalization.

Output Layer: A single neuron with Sigmoid activation to produce a risk probability score between 0 and 1.

Optimization: Trained with the Adam optimizer and Binary Crossentropy loss function, supported by EarlyStopping to halt training when validation loss stops improving.

##  Project Structure
```text
al-fraud-detection-system/
├── src/
│   ├── data_loader.py         # Loads and inspects CSV data
│   ├── eda_manager.py         # Handles initial data visualization
│   ├── preprocess_manager.py  # Data cleaning and scaling
│   ├── model_manager.py       # ANN architecture and training
│   └── visualization_manager.py # Model evaluation and reporting
├── main.py                    # Entry point of the application
└── data/
    └── creditcard.csv         # Dataset file


##  Results 

The model is evaluated on a dedicated test set (15% of total data) that it has never seen before.

Risk Scoring: For every transaction, the model generates a Risk Score.

Decision Threshold: A 0.5 threshold is applied to classify transactions.

Final Output Example:
-------------
Transaction Amount: 320$
Risk Score: 0.94
⚠ Fraud Detected
---------------

## Graphics

To ensure the model's reliability, the following visualizations are generated:

Loss & Accuracy Curves: Monitoring training vs. validation performance over epochs to detect underfitting or overfitting.

Confusion Matrix: Providing a clear view of True Positives (caught frauds) vs. False Negatives (missed frauds).

Correlation Heatmap: Identifying which features have the strongest relationship with fraudulent activity.