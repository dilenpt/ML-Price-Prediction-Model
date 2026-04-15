Mobile Price Prediction using Deep Learning
Overview
This project uses Deep Learning to predict the selling price of mobile phones based on technical specifications and hardware features. The model was developed entirely in a Jupyter Notebook (.ipynb) environment and demonstrates the full machine learning workflow from preprocessing to evaluation.
This project highlights skills in:
Machine Learning
Neural Networks
Regression Modeling
Data Analysis
Feature Engineering
Model Evaluation
Objective
To build a predictive model that estimates the price of a mobile phone using features such as:
RAM
Internal Storage
Battery Capacity
Camera Specifications
Processor Features
Screen Size
Device Capabilities
Technologies Used
Python
Jupyter Notebook
TensorFlow / Keras
Scikit-learn
Pandas
NumPy
Matplotlib
Workflow
1. Data Loading
The dataset was imported and explored using Pandas.
import pandas as pd

df = pd.read_csv("mobile_price_dataset.csv")
df.head()
2. Data Preprocessing
The data was prepared by:
Handling missing values
Encoding categorical variables
Splitting features and target
Scaling numerical inputs
X = df.drop("price", axis=1)
y = df["price"]
3. Data Splitting
The dataset was divided into:
Training Set
Validation Set
Test Set
from sklearn.model_selection import train_test_split
4. Deep Learning Model
A neural network regression model was built to learn relationships between phone specs and price.
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
5. Model Training
The model was trained over multiple epochs with validation monitoring and early stopping.
history = model.fit(...)
6. Final Evaluation
The trained model was evaluated on unseen test data.
Results
Test MAE: 4122.02
Test RMSE: 7861.82
Test R²: 0.7998
Interpretation
R² Score = 0.7998
The model explains approximately 80% of the variation in mobile phone prices, showing strong predictive capability.
MAE = 4122
Average prediction error was around 4122 price units.
RMSE = 7861
Some larger errors were present, but overall performance remained strong.
Key Strengths
Good predictive performance
Effective use of deep learning for tabular data
Clean end-to-end notebook workflow
Real-world business relevance
Possible Improvements
Hyperparameter tuning
Feature selection
XGBoost / Random Forest comparison
Ensemble models
Larger dataset
Real-World Applications
This model can be useful for:
E-commerce pricing tools
Used phone resale estimation
Retail market analysis
Product recommendation systems
