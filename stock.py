import sqlite3

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import io
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aapl_price_predictions.joblib')
EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aapl_eval_data.joblib')


class StockDataAnalysis:

    def __init__(self, data):
        self.data = data.copy()
    
    def set_date(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        return self.data
    
    def observations(self):
        close_prices = self.data['Close']  # just the Close prices for all tickers
        sns.set_theme(style='whitegrid')
        
        figure, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=close_prices, ax=ax)
        ax.set_ylabel('CLOSING SHARE PRICE')
        ax.set_xlabel('DATE')
        ax.set_title('CLOSING SHARE PRICES')
 
        buf = io.BytesIO()

        try:
    
            figure.savefig(buf, format = 'png', bbox_inches = 'tight')
            buf.seek(0)
        finally:
            plt.close(figure)

        return buf

    
    def resampling_data(self, new_data): #resampling the data to weekly and monthly intervals
        resampled_weekly = new_data.resample('W').mean()
        resampled_monthly = new_data.resample('ME').mean()

        return f"\nWEEKLY RESAMPLING:\n{resampled_weekly}\n\nMONTHLY RESAMPLING\n{resampled_monthly}\n"

    def moving_averages(self):
    
        data_MA20 = self.data['Close']['AAPL'].rolling(20).mean() #creating a 20-day moving average
        data_MA200 = self.data['Close']['AAPL'].rolling(200).mean() #creating the 200-day moving average

        plt.figure(figsize = (12, 8))
        plt.plot(data_MA20, label = '20-DAY MOVING AVERAGE')
        plt.plot(data_MA200, label = '200-DAY MOVING AVERAGE')

        plt.xlabel('VOLUME TRADED')
        plt.ylabel('SHARE PRICE')
        plt.title('20-DAY & 200-DAY MOVING AVERAGE')
        plt.tight_layout()

        buf2 = io.BytesIO()

        try:
            plt.savefig(buf2, format = 'png', bbox_inches = 'tight')
            buf2.seek(0)

        finally:
            plt.close()

        return buf2
        
    
    def price_prediction_model(self, data):

        data = data['Close']['AAPL'].copy()
        data = data.to_frame(name='Close')
        data['tomorrow'] = data.shift(-1)
        data = data.dropna()
        X = data
        y = data['tomorrow']
        self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(
        X, 
        y,
        test_size = 0.2,
        shuffle = False)

        self.pipeline = Pipeline([
            ('scalar', StandardScaler()),
            ('model', LinearRegression())
        ])
        self.pipeline.fit(self.X_train, self.y_train)
        self.predictions = self.pipeline.predict(self.X_test)
        return self.predictions
    
    def actualvspredicted_results(self, actual, predicted):

        results = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted
        })
        return results
    
    def r_score(self, y_test, predictions):
        r2 = r2_score(y_test, predictions)
        return r2

if __name__ == '__main__':

    try:
        data = yf.download(["AAPL", "MSFT", "GOOGL", "TSLA"],
                            start="2020-01-01",
                            end="2025-01-01").reset_index()
    except Exception as e:  # noqa: BLE001
        print(f'The {e} error occured!')
 
    new_data = yf.download(["AAPL", "MSFT", "GOOGL", "TSLA"],
                            start="2020-01-01",
                            end="2025-01-01")
 
    stock_data = StockDataAnalysis(data)
    stock_data.set_date()
    predictions = stock_data.price_prediction_model(new_data)  # pass the data in!

    eval_data = {'y_test': stock_data.y_test, 'predictions': stock_data.predictions}
 
    joblib.dump(predictions, MODEL_PATH)

    eval_data = {'y_test': stock_data.y_test, 'predictions': stock_data.predictions}
    joblib.dump(eval_data, EVAL_PATH)

    print(f"Predictions saved to {MODEL_PATH}")
    print(f"Eval data saved to {EVAL_PATH}")