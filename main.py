#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.classification import ClassificationModel
from src.visualization import DataVisualizer

def create_directories():
    """Tạo các thư mục cần thiết"""
    dirs = [
        'data/raw', 
        'data/processed', 
        'results/figures', 
        'results/models',
        'results/csv'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Đã tạo các thư mục cần thiết")

def save_processed_data(X_train, X_test, y_train, y_test, preprocessor):
    """Lưu dữ liệu đã xử lý"""
    print("\n" + "="*50)
    print("LƯU DỮ LIỆU ĐÃ XỬ LÝ")
    print("="*50)
    
    # Lưu dưới dạng CSV
    pd.DataFrame(X_train).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)
    
    # Lưu feature names
    if hasattr(preprocessor, 'expected_columns'):
        pd.Series(preprocessor.expected_columns).to_csv('data/processed/feature_names.csv', index=False)
    
    # Lưu preprocessor
    joblib.dump(preprocessor, 'data/processed/preprocessor.pkl')
    
    print("✅ Đã lưu:")
    print("   - X_train.csv, X_test.csv")
    print("   - y_train.csv, y_test.csv")
    print("   - feature_names.csv")
    print("   - preprocessor.pkl")

def save_results(clf, X_train_processed, y_train, y_test):
    """Lưu kết quả và figures"""
    print("\n" + "="*50)
    print("LƯU KẾT QUẢ")
    print("="*50)
    
    # 1. Lưu kết quả so sánh models
    results_df = pd.DataFrame({
        name: results['metrics'] for name, results in clf.results.items()
    }).T
    results_df = results_df.sort_values('F1-Score', ascending=False)
    results_df.to_csv('results/csv/classification_results.csv')
    print("✅ Đã lưu: results/csv/classification_results.csv")
    
    # 2. Lưu từng model
    for name, results in clf.results.items():
        model_path = f'results/models/{name.replace(" ", "_")}.pkl'
        joblib.dump(results['model'], model_path)
        print(f"✅ Đã lưu model: {model_path}")
    
    # 3. Lưu predictions
    predictions_df = pd.DataFrame()
    for name, results in clf.results.items():
        predictions_df[f'{name}_pred'] = results['predictions']
        if results['probabilities'] is not None:
            predictions_df[f'{name}_proba'] = results['probabilities']
    predictions_df['actual'] = y_test
    predictions_df.to_csv('results/csv/predictions.csv', index=False)
    print("✅ Đã lưu: results/csv/predictions.csv")
    
    # 4. Tạo summary report
    summary = []
    summary.append("="*60)
    summary.append("SUMMARY REPORT - DATA MINING PROJECT")
    summary.append("="*60)
    summary.append(f"\nNgày: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"\nTổng số mẫu: {len(y_train) + len(y_test)}")
    summary.append(f"Train: {len(y_train)} samples")
    summary.append(f"Test: {len(y_test)} samples")
    summary.append(f"Số features: {X_train_processed.shape[1]}")
    summary.append("\n" + "="*60)
    summary.append("KẾT QUẢ PHÂN LỚP")
    summary.append("="*60)
    summary.append(results_df.round(4).to_string())
    summary.append("\n" + "="*60)
    summary.append("TOP 3 MODELS")
    summary.append("="*60)
    
    top3 = results_df.head(3)
    for idx, (model, row) in enumerate(top3.iterrows(), 1):
        summary.append(f"\n{idx}. {model}")
        summary.append(f"   - F1-Score: {row['F1-Score']:.4f}")
        summary.append(f"   - Accuracy: {row['Accuracy']:.4f}")
        summary.append(f"   - Precision: {row['Precision']:.4f}")
        summary.append(f"   - Recall: {row['Recall']:.4f}")
        summary.append(f"   - AUC-ROC: {row['AUC-ROC']:.4f}")
        summary.append(f"   - Train Time: {row['Train Time (s)']:.2f}s")
    
    # Lưu summary
    with open('results/summary_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    print("✅ Đã lưu: results/summary_report.txt")
    
    # 5. Tạo visualization
    print("\n" + "="*50)
    print("TẠO BIỂU ĐỒ")
    print("="*50)
    
    visualizer = DataVisualizer(save_dir='results/figures')
    visualizer.save_all_figures(clf, X_train_processed, y_train, y_test)
    
    return results_df

def run_classification_pipeline():
    """Chạy pipeline phân lớp"""
    print("\n" + "="*80)
    print("PHẦN 1: PHÂN LỚP (CLASSIFICATION)")
    print("="*80)
    
    # 1. LOAD DỮ LIỆU
    loader = DataLoader('data/raw/online_shoppers_intention.csv')
    df = loader.load_data()
    
    # 2. CHIA TRAIN/TEST
    X_train, X_test, y_train, y_test = loader.split_data(test_size=0.3, random_state=42)
    
    # 3. TIỀN XỬ LÝ
    print("\n" + "="*60)
    print("TIỀN XỬ LÝ DỮ LIỆU")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    
    # Fit-transform trên training data
    train_df = pd.concat([X_train, y_train], axis=1)
    X_train_processed, y_train_processed = preprocessor.fit_transform(train_df)
    X_train_processed, y_train_processed = preprocessor.handle_imbalance(X_train_processed, y_train_processed)
    
    # Transform trên test data
    test_df = pd.concat([X_test, y_test], axis=1)
    X_test_processed, y_test_processed = preprocessor.transform(test_df)
    
    print(f"\nKết quả tiền xử lý:")
    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")
    
    # 4. LƯU DỮ LIỆU ĐÃ XỬ LÝ
    save_processed_data(X_train_processed, X_test_processed, 
                        y_train_processed, y_test_processed, preprocessor)
    
    # 5. PHÂN LỚP
    clf = ClassificationModel(X_train_processed, X_test_processed, 
                              y_train_processed, y_test_processed)
    clf.define_models()
    clf.train_and_evaluate(cv=5)
    
    # 6. SO SÁNH KẾT QUẢ
    results_df = clf.compare_models()
    
    # 7. LƯU KẾT QUẢ VÀ FIGURES
    save_results(clf, X_train_processed, y_train_processed, y_test_processed)
    
    return clf

def main():
    """Hàm chính"""
    print("="*80)
    print("DATA MINING PROJECT: DỰ ĐOÁN KHẢ NĂNG MUA HÀNG")
    print("="*80)
    
    # Tạo thư mục
    create_directories()
    
    try:
        # Chạy classification pipeline
        clf = run_classification_pipeline()
        
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH PROJECT")
        print("="*80)
        print("\nKết quả được lưu trong các thư mục:")
        print("   - data/processed/: Dữ liệu đã xử lý")
        print("   - results/csv/: Các file CSV kết quả")
        print("   - results/figures/: Các biểu đồ")
        print("   - results/models/: Các model đã train")
        print("   - results/summary_report.txt: Báo cáo tóm tắt")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()