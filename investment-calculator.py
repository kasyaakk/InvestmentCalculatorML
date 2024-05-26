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
