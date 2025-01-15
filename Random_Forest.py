import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
from Stock_data import StockDataProcessor

class StockPredictor:
    def __init__(self, n_estimators: int = 200, max_depth: int = 15):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.processor = StockDataProcessor()
        
    def train_and_predict(self, 
                         symbol: str,
                         prediction_days: int,
                         train_start: str,
                         train_end: str,
                         indicators: List[str] = ['MA', 'RSI', 'MACD', 'VOLUME', 'VOLATILITY']) -> pd.DataFrame:
        """
        Train model on historical data and make predictions
        """
        # Get training data with all indicators
        train_data = self.processor.get_stock_data(
            symbol=symbol,
            start_date=train_start,
            end_date=train_end,
            indicators=indicators
        )
        
        # Prepare data for ML
        X, y = self.processor.prepare_for_ml(target_days=prediction_days)
        
        # Train the model
        self.model.fit(X, y)
        
        # Get prediction probabilities
        pred_proba = self.model.predict_proba(X)
        
        # Create a DataFrame with the index from X
        results = pd.DataFrame(index=X.index)
        
        # Add columns one by one, making sure they align with the index
        results['Prediction_Date'] = X.index
        results['Future_Date'] = results['Prediction_Date'].apply(
            lambda x: x + pd.Timedelta(days=prediction_days)
        )
        results['Current_Close'] = X['Close']
        results['Future_Close'] = X.index.map(train_data['Close'].shift(-prediction_days))
        results['Confidence_Up'] = pred_proba[:, 1]
        results['Actual_Return'] = ((results['Future_Close'] - results['Current_Close']) / results['Current_Close'] * 100)
        
        # Add prediction ranges
        results['Predicted_Range'] = results.apply(
            lambda x: self._get_prediction_range(x['Confidence_Up']), axis=1
        )
        
        # Clean up results and sort by prediction date
        results = results.dropna()
        results = results.sort_values('Prediction_Date')
        
        return results
    
    def _get_prediction_range(self, confidence: float) -> str:
        """Convert model confidence into prediction range"""
        if confidence >= 0.75:
            return "Strong Gain (>10%)"
        elif confidence >= 0.6:
            return "Moderate Gain (5-10%)"
        elif confidence >= 0.4:
            return "Neutral (-5% to +5%)"
        elif confidence >= 0.25:
            return "Moderate Loss (-10% to -5%)"
        else:
            return "Strong Loss (>-10%)"
        
    def evaluate_predictions(self, results: pd.DataFrame) -> Dict:
        """Calculate prediction accuracy metrics"""
        def check_prediction_accuracy(row):
            pred_range = row['Predicted_Range']
            actual_return = row['Actual_Return']
            
            if pred_range == "Strong Gain (>10%)" and actual_return > 10:
                return True
            elif pred_range == "Moderate Gain (5-10%)" and 5 <= actual_return <= 10:
                return True
            elif pred_range == "Neutral (-5% to +5%)" and -5 <= actual_return <= 5:
                return True
            elif pred_range == "Moderate Loss (-10% to -5%)" and -10 <= actual_return <= -5:
                return True
            elif pred_range == "Strong Loss (>-10%)" and actual_return < -10:
                return True
            return False
        
        results['Prediction_Correct'] = results.apply(check_prediction_accuracy, axis=1)
        
        metrics = {
            'Overall_Accuracy': results['Prediction_Correct'].mean(),
            'Prediction_Counts': results['Predicted_Range'].value_counts().to_dict(),
            'Average_Return': results['Actual_Return'].mean()
        }
        
        return metrics

# Example usage
if __name__ == "__main__":
    predictor = StockPredictor(n_estimators=200, max_depth=15)
    
    # Train on data from 2010 to 2020
    results = predictor.train_and_predict(
        symbol='^GSPC',
        prediction_days=365,
        train_start='2010-01-01',
        train_end='2021-01-01'
    )
    
    # Evaluate predictions
    metrics = predictor.evaluate_predictions(results)
    
    print("\nPrediction Results:")
    print(f"Overall Accuracy: {metrics['Overall_Accuracy']:.2%}")
    print("\nPrediction Distribution:")
    for range_name, count in metrics['Prediction_Counts'].items():
        print(f"{range_name}: {count}")
    print(f"\nAverage Return: {metrics['Average_Return']:.2f}%")
    
    # Display some example predictions
    print("\nSample Predictions:")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(results[['Prediction_Date', 'Future_Date', 'Current_Close', 'Predicted_Range', 'Actual_Return']].tail(10))
