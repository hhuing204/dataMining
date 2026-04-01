import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

class ClassificationModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        self.best_model = None
        
        # Đảm bảo dữ liệu là numpy array để tránh lỗi pandas
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        
    def define_models(self):
        """Định nghĩa các mô hình cần thử nghiệm"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        print(f"Đã định nghĩa {len(self.models)} mô hình")
        return self
    
    def train_and_evaluate(self, cv=5):
        """Train và đánh giá tất cả mô hình"""
        print("\n" + "="*80)
        print("TRAIN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH")
        print("="*80)
        
        # Kiểm tra dữ liệu đầu vào
        print(f"\nKiểm tra dữ liệu:")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_train dtype: {self.X_train.dtype}")
        print(f"Mẫu X_train[0]: {self.X_train[0][:5]}...")  # 5 giá trị đầu
        
        for name, model in self.models.items():
            print(f"\n>>> Đang train: {name}...")
            
            try:
                # Train time
                start_time = time.time()
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                           cv=cv, scoring='f1')
                
                # Train trên toàn bộ train set
                model.fit(self.X_train, self.y_train)
                
                # Predict
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Thời gian train
                train_time = time.time() - start_time
                
                # Tính metrics
                metrics = {
                    'CV Mean F1': cv_scores.mean(),
                    'CV Std F1': cv_scores.std(),
                    'Accuracy': accuracy_score(self.y_test, y_pred),
                    'Precision': precision_score(self.y_test, y_pred),
                    'Recall': recall_score(self.y_test, y_pred),
                    'F1-Score': f1_score(self.y_test, y_pred),
                    'AUC-ROC': roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None,
                    'Train Time (s)': train_time
                }
                
                self.results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✓ F1-Score: {metrics['F1-Score']:.4f}")
                print(f"  ✓ AUC-ROC: {metrics['AUC-ROC']:.4f}")
                print(f"  ✓ Train Time: {train_time:.2f}s")
                
            except Exception as e:
                print(f"  ✗ LỖI: {e}")
                continue
        
        return self
    
    def compare_models(self):
        """So sánh các mô hình"""
        if not self.results:
            print("Không có kết quả nào để so sánh")
            return pd.DataFrame()
        
        results_df = pd.DataFrame({
            name: results['metrics'] for name, results in self.results.items()
        }).T
        
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*80)
        print("SO SÁNH CÁC MÔ HÌNH")
        print("="*80)
        print(results_df.round(4))
        
        return results_df