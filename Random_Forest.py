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
from sklearn.metrics import confusion_matrix



class StockPredictor:
    def __init__(self, n_estimators: int = 200, max_depth: int = 15):
        pass

# Example usage
if __name__ == "__main__":
    processor = StockDataProcessor()
    data = processor.get_stock_data("^GSPC")
    data, target = processor.prepare_for_ml(100)
    df = pd.DataFrame(data)
    df['target'] = target
    print(f"Data:\n {df}")

    x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), target, test_size=0.2)
    model = RandomForestClassifier(n_estimators=40)
    model.fit(x_train, y_train)

    print(model.score(x_test, y_test))

    y_predicted = model.predict(x_test)
    cm = confusion_matrix(y_test, y_predicted)
    print(cm)
