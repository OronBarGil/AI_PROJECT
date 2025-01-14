import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.ensemble import RandomForestClassifier


class StockDataProcessor:
    def __init__(self):
        """Initialize processor with empty data"""
        self.data = None
        self.available_indicators = {
            'MA': self._calculate_ma, #moving average - average price of the stock
            'RSI': self._calculate_rsi, #relative strength index - used to decide if a stock is overpriced or underpriced
            'MACD': self._calculate_macd, #moving average convergence divergence - used for measuring a stock's momentum
            'VOLUME': self._calculate_volume_indicators, #the number of shares traded in a stock 
            'VOLATILITY': self._calculate_volatility #describes when a market experienced periods of unpredictable, and sometimes sharp, price movements

        }
    def get_stock_data(self, 
                      symbol: str,
                      start_date: str = '2010-01-01',
                      end_date: Optional[str] = None,
                      indicators: List[str] = None,
                      ma_periods: List[int] = [50, 100, 150]) -> pd.DataFrame:
        """
        Fetch stock data and calculate requested indicators
        
        Args:
            symbol: Stock/Index symbol (e.g., '^GSPC' for S&P 500)
            start_date: Start date for historical data
            end_date: End date for historical data
            indicators: List of indicators to calculate ['MA', 'RSI', 'MACD', 'VOLUME', 'VOLATILITY']
            ma_periods: List of periods for moving averages
        """
        # Initialize ticker
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        self.data = ticker.history(start=start_date, end=end_date)
        
        # Calculate requested indicators
        if indicators:
            for indicator in indicators:
                if indicator in self.available_indicators:
                    if indicator == 'MA':
                        self.available_indicators[indicator](periods=ma_periods)
                    else:
                        self.available_indicators[indicator]()
                else:
                    print(f"Warning: Indicator {indicator} not found")
        
        return self.data

    def _calculate_ma(self, periods: List[int] = [50, 100, 150]) -> None:
        """Calculate Moving Averages for specified periods"""
        for period in periods:
            self.data[f'MA{period}'] = self.data['Close'].rolling(window=period).mean()

    def _calculate_rsi(self, period: int = 14) -> None:
        """Calculate Relative Strength Index"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

    def _calculate_macd(self) -> None:
        """Calculate MACD and Signal Line"""
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()

    def _calculate_volume_indicators(self, periods: List[int] = [50, 100, 150]) -> None:
        """Calculate Volume-based indicators"""
        for period in periods:
            self.data[f'Volume_MA{period}'] = self.data['Volume'].rolling(window=period).mean()
        self.data['Volume_Change'] = self.data['Volume'].pct_change()

    def _calculate_volatility(self, period: int = 20) -> None:
        """Calculate Volatility indicators"""
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Daily_Return'].rolling(window=period).std()

    def prepare_for_ml(self, target_days: int = 1) -> tuple:
        """
        Prepare data for machine learning
        
        Args:
            target_days: Number of days ahead to predict
            
        Returns:
            tuple: (X, y) features and target
        """
        # Create target variable (future price movement)
        self.data['Target'] = self.data['Close'].shift(-target_days) > self.data['Close']
        
        # Drop NaN values
        self.data = self.data.dropna()
        
        # Separate features and target
        feature_columns = [col for col in self.data.columns if col not in ['Target', 'Dividends', 'Stock Splits']]
        X = self.data[feature_columns]
        y = self.data['Target']
        
        return X, y

    
# Example usage
if __name__ == "__main__":
    pass


