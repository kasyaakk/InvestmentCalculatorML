import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
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
    lr_model_eur_to_usd = LinearRegression()
    lr_model_eur_to_usd.fit(X, y_eur_to_usd)

    dt_model_eur_to_usd = DecisionTreeRegressor()
    dt_model_eur_to_usd.fit(X, y_eur_to_usd)

    lr_model_inflation_rate = LinearRegression()
    lr_model_inflation_rate.fit(X, y_inflation_rate)

    dt_model_inflation_rate = DecisionTreeRegressor()
    dt_model_inflation_rate.fit(X, y_inflation_rate)

    return (lr_model_eur_to_usd, dt_model_eur_to_usd), (lr_model_inflation_rate, dt_model_inflation_rate)

# Function to make predictions
def make_predictions(models, future_dates_num):
    (lr_model_eur_to_usd, dt_model_eur_to_usd), (lr_model_inflation_rate, dt_model_inflation_rate) = models

    lr_predicted_eur_to_usd = lr_model_eur_to_usd.predict(future_dates_num)
    dt_predicted_eur_to_usd = dt_model_eur_to_usd.predict(future_dates_num)
    
    lr_predicted_inflation_rate = lr_model_inflation_rate.predict(future_dates_num)
    dt_predicted_inflation_rate = dt_model_inflation_rate.predict(future_dates_num)

    return (lr_predicted_eur_to_usd, dt_predicted_eur_to_usd), (lr_predicted_inflation_rate, dt_predicted_inflation_rate)

# Function to calculate financial prospects
def calculate_financial_prospects(initial_investment, additional_investment, frequency, years, rate_of_return, compound_frequency, predicted_eur_to_usd, predicted_inflation_rate):
    periods = {
        'daily': 365,
        'monthly': 12,
        'annually': 1
    }[compound_frequency]

    future_value = initial_investment
    investment_values = []

    for i in range(len(predicted_eur_to_usd)):
        if frequency == 'monthly':
            if i % 1 == 0:
                future_value += additional_investment
        elif frequency == 'annually':
            if i % 12 == 0:
                future_value += additional_investment

        future_value *= (1 + rate_of_return / periods)
        future_value *= (1 + predicted_inflation_rate[i])
        future_value_usd = future_value * predicted_eur_to_usd[i]
        investment_values.append(future_value_usd)

    return investment_values

# Function to visualize data
def visualize(data, future_dates, lr_predicted_eur_to_usd, dt_predicted_eur_to_usd, lr_predicted_inflation_rate, dt_predicted_inflation_rate, lr_financial_prospects, dt_financial_prospects):
    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(data['date'], data['eur_to_usd'], label='Historical EUR/USD Exchange Rate')
    plt.plot(future_dates, lr_predicted_eur_to_usd, label='Predicted EUR/USD Exchange Rate (Linear Regression)')
    plt.plot(future_dates, dt_predicted_eur_to_usd, label='Predicted EUR/USD Exchange Rate (Decision Tree)', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('EUR/USD')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(data['date'], data['inflation_rate'], label='Historical Inflation Rate')
    plt.plot(future_dates, lr_predicted_inflation_rate, label='Predicted Inflation Rate (Linear Regression)')
    plt.plot(future_dates, dt_predicted_inflation_rate, label='Predicted Inflation Rate (Decision Tree)', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Inflation Rate')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(future_dates, lr_financial_prospects, label='Predicted Financial Prospects (in USD, Linear Regression)')
    plt.plot(future_dates, dt_financial_prospects, label='Predicted Financial Prospects (in USD, Decision Tree)', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Amount (in USD)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(future_dates, np.array(lr_financial_prospects) - np.array(dt_financial_prospects), label='Financial Difference (Linear Regression vs Decision Tree)')
    plt.xlabel('Date')
    plt.ylabel('Difference in Amount (in USD)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # Requesting user input
    initial_investment = float(input("Enter initial investment amount (in euros): "))
    additional_investment = float(input("Enter additional investment amount (in euros): "))
    frequency = input("Enter investment frequency (monthly/annually): ")
    years = int(input("Enter investment duration (in years): "))
    rate_of_return = float(input("Enter expected annual rate of return (in decimal, e.g., 0.05 for 5%): "))
    compound_frequency = input("Enter compound frequency (daily/monthly/annually): ")
    start_date_str = input("Enter start date (in YYYY-MM-DD format): ")
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    data = load_data()
    X, y_eur_to_usd, y_inflation_rate = prepare_data(data)
    models = train_models(X, y_eur_to_usd, y_inflation_rate)

    future_dates = pd.date_range(start=start_date, periods=years*12, freq='M')
    future_dates_num = future_dates.to_series().apply(lambda x: x.toordinal()).values.reshape(-1, 1)

    (lr_predicted_eur_to_usd, dt_predicted_eur_to_usd), (lr_predicted_inflation_rate, dt_predicted_inflation_rate) = make_predictions(models, future_dates_num)

    lr_financial_prospects = calculate_financial_prospects(initial_investment, additional_investment, frequency, years, rate_of_return, compound_frequency, lr_predicted_eur_to_usd, lr_predicted_inflation_rate)
    dt_financial_prospects = calculate_financial_prospects(initial_investment, additional_investment, frequency, years, rate_of_return, compound_frequency, dt_predicted_eur_to_usd, dt_predicted_inflation_rate)

    visualize(data, future_dates, lr_predicted_eur_to_usd, dt_predicted_eur_to_usd, lr_predicted_inflation_rate, dt_predicted_inflation_rate, lr_financial_prospects, dt_financial_prospects)

# Running the main function
main()
