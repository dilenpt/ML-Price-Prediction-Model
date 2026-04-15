# Mobile Price Prediction using Deep Learning

## Overview

This project uses **Deep Learning** to predict the selling price of mobile phones based on technical specifications and hardware features. The model was developed entirely in a **Jupyter Notebook (.ipynb)** environment and demonstrates the full machine learning workflow from preprocessing to evaluation.

This project highlights skills in:

- Machine Learning  
- Neural Networks  
- Regression Modeling  
- Data Analysis  
- Feature Engineering  
- Model Evaluation  

## Objective

To build a predictive model that estimates the price of a mobile phone using features such as:

- RAM  
- Internal Storage  
- Battery Capacity  
- Camera Specifications  
- Processor Features  
- Screen Size  
- Device Capabilities  

## Technologies Used

- Python
- Jupyter Notebook
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Workflow

### 1. Data Loading

The dataset was imported and explored using Pandas.

```python
import pandas as pd
df = pd.read_csv("mobile_price_dataset.csv")
df.head()
```

### 2. Data Preprocessing

- Handling missing values
- Encoding categorical variables
- Splitting features and target
- Scaling numerical inputs

```python
X = df.drop("price", axis=1)
y = df["price"]
```

### 3. Data Splitting

- Training Set
- Validation Set
- Test Set

### 4. Deep Learning Model

```python
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
```

### 5. Final Evaluation

```text
Test MAE: 4122.02
Test RMSE: 7861.82
Test R²: 0.7998
```

## Interpretation

- **R² = 0.7998** → Explains about 80% of price variation.
- **MAE = 4122** → Average prediction error around 4122 price units.
- **RMSE = 7861** → Some larger errors present.

## Folder Structure

```text
Mobile_Price_Prediction.ipynb
README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow jupyter
jupyter notebook
```

Open:

```text
Mobile_Price_Prediction.ipynb
```

Run all cells in order.

## Author

Dilen Patel

Master’s Student in Computer Science  
Interested in Machine Learning, AI, and Predictive Analytics
