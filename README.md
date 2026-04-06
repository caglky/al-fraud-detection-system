## AI-Based Credit Card Fraud Detection System

This project is a simple machine learning applications that aims to detect fradulent credit card transactions. The goal is to understand how machine learning models can be used to solve real-world problems such as fraud detection.

## Problem 

Credit card fraud is a series issue in financial systems. One of the main challenges is that fraud cases are very rare compared to normal transactions. This project focuses on building a model that can distinguish between normal and fradulent transactions. 

---

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
```

---

## Dataset
The dataset contains transactions made by European cardholders.

- Features : 28 anonymized variables (V1-V28), plus Time and Amount
- Target  : Class (0 : Normal, 1 : Fraud)

---

##  What I Did
- Performed basic data analysis (EDA)
- Scaled the data using StandardScaler
- Split the dataset into training and test sets
- Build a simple Neural Network model
- Evaluated the model using accuracy and loss
- Visualized results (loss curevesi confusion matrix)

---

## Model 
I used a simple neural network with:
- Input layer matching feature size
- A few Dense layers with ReLu activation
- Dropout layers to reduce overfitting
- Output layer with Sigmoid activation

---


##  Results 

The model is tested on unseen data:
- Predicts whether a transaction is fraud or not
- Outputs a probability score
- Uses a threshold (0.5) for classification

---

## Visualizations
- Fraud Distribution
- Transaction Amount Distribution
- Time vs Process Density
- Correlation 
- Model Loss Curve and Model Accuracy Curve
- Confusion Matrix

--- 
## Motivation 
This project was build to practice:
- Machine learning basics
- Data preprocessing
- Neural Networks
- Working with real datasets

--- 
## Author
Build as a learning project while studying computer engineering.
