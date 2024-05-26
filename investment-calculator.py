import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Function to load data 
def load_data():
    current_date = datetime.now()
    data = pd.DataFrame({
        'date': pd.date_range(start=current_date - timedelta(days=365*10), periods=120, freq='M'),
        'eur_to_usd': np.random.uniform(1.1, 1.2, 120),
        'inflation_rate': np.random.uniform(0.01, 0.03, 120)
    })
    return data

# Function to prepare data
def prepare_data(data):
    data['date_num'] = data['date'].apply(lambda x: x.toordinal())
    X = data[['date_num']]
    y_eur_to_usd = data['eur_to_usd']
    y_inflation_rate = data['inflation_rate']
    return X, y_eur_to_usd, y_inflation_rate

# Function to train models
def train_models(X, y_eur_to_usd, y_inflation_rate):
    model_eur_to_usd = LinearRegression()
    model_eur_to_usd.fit(X, y_eur_to_usd)
    model_inflation_rate = LinearRegression()
    model_inflation_rate.fit(X, y_inflation_rate)
    return model_eur_to_usd, model_inflation_rate
