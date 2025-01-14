import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Dict
from Stock_data import StockDataProcessor

class StockPredictor:
    def __init__(self, n_estimators: int = 200, max_depth: int = 15):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.processor = StockDataProcessor()
        self.feature_importance = None
        self.training_data = None
        self.validation_data = None

    def train_and_validate(self, 
                          symbol: str,
                          train_start: str = '2010-01-01',
                          train_end: str = '2020-01-01',
                          validation_end: str = '2020-12-30',  # Current date
                          target_days: int = 1) -> Dict:
        """
        Train on historical data and validate on recent data
        """
        # Get full dataset
        self.processor.get_stock_data(
            symbol=symbol,
            start_date=train_start,
            end_date=validation_end,
            indicators=['MA', 'RSI', 'MACD', 'VOLUME', 'VOLATILITY']
        )
        
        # Prepare features and target
        X, y = self.processor.prepare_for_ml(target_days=target_days)
        feature_names = X.columns
        
        # Split data into training and validation periods
        train_mask = (X.index >= train_start) & (X.index <= train_end)
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store training data for reference
        self.training_data = {
            'X_train': X_train,
            'y_train': y_train,
            'feature_names': feature_names
        }
        
        # Validate on post-training period
        validation_mask = (X.index > train_end) & (X.index <= validation_end)
        X_val = X[validation_mask]
        y_val = y[validation_mask]
        X_val_scaled = self.scaler.transform(X_val)
        
        # Make validation predictions
        val_predictions = self.model.predict(X_val_scaled)
        
        # Store validation results
        self.validation_data = {
            'actual_prices': self.processor.data['Close'][validation_mask],
            'predictions': val_predictions,
            'actual_movements': y_val
        }
        
        return {
            'training_period': f"{train_start} to {train_end}",
            'validation_period': f"{train_end} to {validation_end}",
            'validation_report': classification_report(y_val, val_predictions),
            'feature_importance': self.feature_importance
        }

    def predict_future(self, 
                      symbol: str, 
                      target_date: str = '2020-12-30',
                      simulation_runs: int = 1000) -> Dict:
        """
        Predict future price movements using Monte Carlo simulation
        """
        # Get latest data
        latest_data = self.processor.get_stock_data(
            symbol=symbol,
            start_date=(pd.Timestamp.now() - pd.Timedelta(days=200)).strftime('%Y-%m-%d'),
            indicators=['MA', 'RSI', 'MACD', 'VOLUME', 'VOLATILITY']
        )
        
        # Current price
        current_price = latest_data['Close'].iloc[-1]
        
        # Calculate daily returns volatility
        daily_returns = latest_data['Close'].pct_change().dropna()
        volatility = daily_returns.std()
        
        # Number of days to simulate
        days_to_simulate = (pd.Timestamp(target_date) - pd.Timestamp.now()).days
        
        # Monte Carlo simulation
        simulated_prices = []
        for _ in range(simulation_runs):
            price = current_price
            price_path = [price]
            
            for _ in range(days_to_simulate):
                # Generate random return
                return_rate = np.random.normal(daily_returns.mean(), volatility)
                price = price * (1 + return_rate)
                price_path.append(price)
            
            simulated_prices.append(price_path[-1])
        
        # Calculate prediction statistics
        predicted_price = np.mean(simulated_prices)
        confidence_interval = np.percentile(simulated_prices, [5, 95])
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'confidence_interval': confidence_interval,
            'prediction_date': target_date,
            'up_probability': np.mean(np.array(simulated_prices) > current_price) * 100
        }

# Example usage
if __name__ == "__main__":
     # Initialize predictor
    predictor = StockPredictor(n_estimators=200, max_depth=15)
    
    # Choose stock symbol (S&P 500)
    symbol = '^GSPC'
    
    print("Training model on 2020-2022 data...")
    results = predictor.train_and_validate(
        symbol=symbol,
        train_start='2020-01-01',
        train_end='2022-12-31'
    )
    
    print("\n=== Training Performance ===")
    print(f"Training Period: {results['training_period']}")
    print(f"Validation Period: {results['validation_period']}")
    print("\nValidation Report:")
    print(results['validation_report'])
    
    # Make prediction for end of 2024
    future_prediction = predictor.predict_future(
        symbol=symbol,
        target_date='2024-12-31'
    )
    
    print("\n=== S&P 500 Price Prediction for December 31, 2024 ===")
    print(f"Current Price (As of Jan 2024): ${future_prediction['current_price']:,.2f}")
    print(f"Predicted Price (Dec 31, 2024): ${future_prediction['predicted_price']:,.2f}")
    print("\nPrediction Details:")
    print(f"- Price Change: ${(future_prediction['predicted_price'] - future_prediction['current_price']):,.2f}")
    print(f"- Percentage Change: {((future_prediction['predicted_price'] / future_prediction['current_price'] - 1) * 100):,.1f}%")
    print(f"- 90% Confidence Range: ${future_prediction['confidence_interval'][0]:,.2f} to ${future_prediction['confidence_interval'][1]:,.2f}")
    print(f"- Probability of Increase: {future_prediction['up_probability']:.1f}%")