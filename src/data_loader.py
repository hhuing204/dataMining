import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import urllib.request

class DataLoader:
    def __init__(self, data_path='data/raw/online_shoppers_intention.csv'):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load dữ liệu từ file CSV"""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        if not os.path.exists(self.data_path):
            print(f"Đang tải dữ liệu từ UCI...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
            self.df = pd.read_csv(url)
            self.df.to_csv(self.data_path, index=False)
            print("Đã tải và lưu dữ liệu")
        else:
            self.df = pd.read_csv(self.data_path)
            print(f"Đã load dữ liệu từ file local")
        
        print(f"Kích thước: {self.df.shape}")
        return self.df
    
    def split_data(self, test_size=0.3, random_state=42):
        """Chia dữ liệu train/test"""
        if self.df is None:
            self.load_data()
            
        X = self.df.drop('Revenue', axis=1)
        y = self.df['Revenue']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nCHIA DỮ LIỆU:")
        print(f"Train: {len(X_train)} mẫu ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Test: {len(X_test)} mẫu ({len(X_test)/len(X)*100:.1f}%)")
        print(f"Train distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Test distribution:\n{y_test.value_counts(normalize=True)}")
        
        return X_train, X_test, y_train, y_test