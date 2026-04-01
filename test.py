import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_preprocessor():
    """Kiểm tra preprocessor với dữ liệu thật"""
    
    print("="*80)
    print("TEST PREPROCESSOR")
    print("="*80)
    
    # Load data
    loader = DataLoader('data/raw/online_shoppers_intention.csv')
    df = loader.load_data()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    
    # Fit on train
    train_df = pd.concat([X_train, y_train], axis=1)
    X_train_processed, y_train_processed = preprocessor.fit_transform(train_df)
    
    # Transform on test
    test_df = pd.concat([X_test, y_test], axis=1)
    X_test_processed, y_test_processed = preprocessor.transform(test_df)
    
    # Check shapes
    print(f"\n--- KIỂM TRA SHAPE ---")
    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")
    
    # Check columns
    train_cols = set(X_train_processed.columns)
    test_cols = set(X_test_processed.columns)
    
    if train_cols == test_cols:
        print("\n✅ Train và test có cùng số cột")
    else:
        print(f"\n❌ Số cột không khớp:")
        print(f"   Thiếu trong test: {train_cols - test_cols}")
        print(f"   Dư trong test: {test_cols - train_cols}")
    
    # Check 'Weekend' column specifically
    if 'Weekend' in X_train_processed.columns:
        print(f"\n--- KIỂM TRA CỘT WEEKEND ---")
        print(f"Weekend trong train: {X_train_processed['Weekend'].unique()}")
        print(f"Weekend trong test: {X_test_processed['Weekend'].unique()}")
    
    # Train model
    print(f"\n--- TRAIN MODEL ---")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train_processed, y_train_processed)
    
    # Predict
    y_pred = model.predict(X_test_processed)
    acc = accuracy_score(y_test_processed, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    
    return preprocessor

def test_with_missing_weekend():
    """Test trường hợp test data thiếu cột Weekend"""
    
    print("\n" + "="*80)
    print("TEST TRƯỜNG HỢP THIẾU WEEKEND")
    print("="*80)
    
    # Load data
    loader = DataLoader('data/raw/online_shoppers_intention.csv')
    df = loader.load_data()
    
    # Tạo train data có Weekend
    train_data = df.iloc[:8000].copy()
    
    # Tạo test data KHÔNG có Weekend
    test_data = df.iloc[8000:10000].copy()
    test_data = test_data.drop('Weekend', axis=1)
    
    print(f"\nTrain columns: {train_data.columns.tolist()}")
    print(f"Test columns: {test_data.columns.tolist()}")
    print(f"Weekend trong train? {'Weekend' in train_data.columns}")
    print(f"Weekend trong test? {'Weekend' in test_data.columns}")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    
    # Fit on train
    X_train_processed, y_train_processed = preprocessor.fit_transform(train_data)
    
    # Transform on test
    X_test_processed, y_test_processed = preprocessor.transform(test_data)
    
    # Check if Weekend exists in processed test
    print(f"\n--- KẾT QUẢ ---")
    print(f"Weekend trong X_test_processed? {'Weekend' in X_test_processed.columns}")
    if 'Weekend' in X_test_processed.columns:
        print(f"Giá trị Weekend trong test: {X_test_processed['Weekend'].unique()}")
    
    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_processed, y_train_processed)
    y_pred = model.predict(X_test_processed)
    acc = accuracy_score(y_test_processed, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # Test cơ bản
    preprocessor = test_preprocessor()
    
    # Test trường hợp thiếu Weekend
    test_with_missing_weekend()