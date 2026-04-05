import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Xử lý tiền xử lý dữ liệu với khả năng đồng bộ train/test
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.expected_columns = []  # Danh sách các cột mong đợi sau khi xử lý
        self.numeric_columns = []
        self.is_fitted = False
        self.all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.all_visitors = ['New_Visitor', 'Returning_Visitor', 'Other']
        self.month_columns = []
        self.visitor_columns = []
        
    def _create_month_dummies(self, df, prefix='Month'):
        """Tạo one-hot encoding cho Month với đầy đủ các cột"""
        dummies = pd.DataFrame(0, index=df.index, 
                              columns=[f"{prefix}_{m}" for m in self.all_months])
        
        for month in df.unique():
            if month in self.all_months:
                col_name = f"{prefix}_{month}"
                dummies.loc[df == month, col_name] = 1
        
        return dummies
    
    def _create_visitor_dummies(self, df, prefix='VisitorType'):
        """Tạo one-hot encoding cho VisitorType với đầy đủ các cột"""
        dummies = pd.DataFrame(0, index=df.index, 
                              columns=[f"{prefix}_{v}" for v in self.all_visitors])
        
        for visitor in df.unique():
            if visitor in self.all_visitors:
                col_name = f"{prefix}_{visitor}"
                dummies.loc[df == visitor, col_name] = 1
        
        return dummies
    
    def fit_transform(self, df, target_col='Revenue'):
        """
        Fit preprocessor trên training data và transform
        """
        print("\n" + "="*70)
        print("FIT-TRANSFORM TRÊN TRAINING DATA")
        print("="*70)
        
        df = df.copy()
        self.target_col = target_col
        
        # 1. TÁCH FEATURES VÀ TARGET
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = None
            
        print(f"Shape ban đầu: {X.shape}")
        
        # 2. XỬ LÝ TỪNG CỘT
        processed_parts = []
        
        # 2.1. GIỮ LẠI CÁC CỘT SỐ (KHÔNG PHẢI CATEGORICAL)
        numeric_cols = ['Administrative', 'Administrative_Duration', 'Informational', 
                       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                       'OperatingSystems', 'Browser', 'Region', 'TrafficType']
        
        # Lấy các cột số có trong dữ liệu
        existing_numeric = [col for col in numeric_cols if col in X.columns]
        X_numeric = X[existing_numeric].copy()
        print(f"\n--- CÁC CỘT SỐ ---")
        print(f"Đã giữ lại {len(existing_numeric)} cột số")
        
        # 2.2. XỬ LÝ MONTH
        if 'Month' in X.columns:
            print("\n--- XỬ LÝ MONTH ---")
            print(f"Giá trị Month trong train: {sorted(X['Month'].unique())}")
            month_dummies = self._create_month_dummies(X['Month'])
            self.month_columns = month_dummies.columns.tolist()
            print(f"Đã tạo {len(self.month_columns)} cột: {self.month_columns[:5]}...")
            processed_parts.append(month_dummies)
        
        # 2.3. XỬ LÝ VISITORTYPE
        if 'VisitorType' in X.columns:
            print("\n--- XỬ LÝ VISITORTYPE ---")
            print(f"Giá trị VisitorType trong train: {sorted(X['VisitorType'].unique())}")
            visitor_dummies = self._create_visitor_dummies(X['VisitorType'])
            self.visitor_columns = visitor_dummies.columns.tolist()
            print(f"Đã tạo {len(self.visitor_columns)} cột")
            processed_parts.append(visitor_dummies)
        
        # 2.4. XỬ LÝ WEEKEND - QUAN TRỌNG: GIỮ LẠI CỘT NÀY
        if 'Weekend' in X.columns:
            print("\n--- XỬ LÝ WEEKEND ---")
            print(f"Giá trị Weekend trong train: {X['Weekend'].unique()}")
            weekend_processed = pd.DataFrame({
                'Weekend': X['Weekend'].astype(int)
            }, index=X.index)
            print(f"Đã chuyển Weekend sang số: {weekend_processed['Weekend'].unique()}")
            processed_parts.append(weekend_processed)
        
        # 3. KẾT HỢP TẤT CẢ CÁC PHẦN
        if processed_parts:
            X_final = pd.concat([X_numeric] + processed_parts, axis=1)
        else:
            X_final = X_numeric
        
        # 4. XÁC ĐỊNH DANH SÁCH CỘT CUỐI CÙNG
        self.expected_columns = X_final.columns.tolist()
        self.numeric_columns = X_final.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\n--- TỔNG KẾT ---")
        print(f"Tổng số cột sau xử lý: {len(self.expected_columns)}")
        print(f"Các cột cuối cùng: {self.expected_columns}")
        
        # 5. CHUẨN HÓA DỮ LIỆU SỐ
        print("\n--- CHUẨN HÓA DỮ LIỆU ---")
        X_scaled = X_final.copy()
        X_scaled[self.numeric_columns] = self.scaler.fit_transform(X_final[self.numeric_columns])
        print("Đã chuẩn hóa dữ liệu số")
        
        # 6. KIỂM TRA LẦN CUỐI
        print("\n--- KIỂM TRA LẦN CUỐI ---")
        print(f"Shape: {X_scaled.shape}")
        print(f"Các cột: {X_scaled.columns.tolist()}")
        
        self.is_fitted = True
        print("\n" + "="*70)
        print("HOÀN TẤT FIT-TRANSFORM")
        print("="*70)
        
        if y is not None:
            return X_scaled, y
        return X_scaled
    
    def transform(self, df):
        """
        Transform test data sử dụng preprocessor đã fit
        """
        print("\n" + "="*70)
        print("TRANSFORM TRÊN TEST DATA")
        print("="*70)
        
        if not self.is_fitted:
            raise ValueError("Preprocessor chưa được fit. Gọi fit_transform trước.")
        
        df = df.copy()
        
        # 1. TÁCH FEATURES VÀ TARGET
        if self.target_col in df.columns:
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]
        else:
            X = df
            y = None
            
        print(f"Shape ban đầu: {X.shape}")
        
        # 2. XỬ LÝ TỪNG CỘT (GIỐNG HỆT FIT)
        processed_parts = []
        
        # 2.1. CÁC CỘT SỐ
        numeric_cols = ['Administrative', 'Administrative_Duration', 'Informational', 
                       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                       'OperatingSystems', 'Browser', 'Region', 'TrafficType']
        
        existing_numeric = [col for col in numeric_cols if col in X.columns]
        X_numeric = X[existing_numeric].copy()
        
        # 2.2. XỬ LÝ MONTH
        if 'Month' in X.columns:
            month_dummies = self._create_month_dummies(X['Month'])
            # Chỉ giữ lại các cột đã có trong training
            if hasattr(self, 'month_columns'):
                for col in self.month_columns:
                    if col not in month_dummies.columns:
                        month_dummies[col] = 0
                month_dummies = month_dummies[self.month_columns]
            processed_parts.append(month_dummies)
        
        # 2.3. XỬ LÝ VISITORTYPE
        if 'VisitorType' in X.columns:
            visitor_dummies = self._create_visitor_dummies(X['VisitorType'])
            if hasattr(self, 'visitor_columns'):
                for col in self.visitor_columns:
                    if col not in visitor_dummies.columns:
                        visitor_dummies[col] = 0
                visitor_dummies = visitor_dummies[self.visitor_columns]
            processed_parts.append(visitor_dummies)
        
        # 2.4. XỬ LÝ WEEKEND - QUAN TRỌNG: PHẢI CÓ CỘT NÀY
        if 'Weekend' in X.columns:
            weekend_processed = pd.DataFrame({
                'Weekend': X['Weekend'].astype(int)
            }, index=X.index)
            processed_parts.append(weekend_processed)
        elif 'Weekend' in self.expected_columns:
            # Nếu test không có Weekend, tạo cột với giá trị mặc định (0)
            print("\n--- CẢNH BÁO: Test không có cột Weekend, tạo cột mới với giá trị 0 ---")
            weekend_processed = pd.DataFrame({
                'Weekend': np.zeros(len(X), dtype=int)
            }, index=X.index)
            processed_parts.append(weekend_processed)
        
        # 3. KẾT HỢP TẤT CẢ CÁC PHẦN
        if processed_parts:
            X_final = pd.concat([X_numeric] + processed_parts, axis=1)
        else:
            X_final = X_numeric
        
        # 4. ĐẢM BẢO CÓ ĐỦ CÁC CỘT NHƯ TRAINING
        print(f"\n--- ĐỒNG BỘ CỘT VỚI TRAINING ---")
        print(f"Số cột hiện tại: {len(X_final.columns)}")
        print(f"Số cột mong đợi: {len(self.expected_columns)}")
        
        # Tạo DataFrame mới với đúng thứ tự cột
        X_aligned = pd.DataFrame(index=X_final.index)
        
        for col in self.expected_columns:
            if col in X_final.columns:
                X_aligned[col] = X_final[col]
            else:
                print(f"  + Thêm cột thiếu: {col} (giá trị 0)")
                X_aligned[col] = 0
        
        print(f"Số cột sau đồng bộ: {len(X_aligned.columns)}")
        
        # 5. CHUẨN HÓA
        X_scaled = X_aligned.copy()
        X_scaled[self.numeric_columns] = self.scaler.transform(X_aligned[self.numeric_columns])
        print("Đã chuẩn hóa dữ liệu số")
        
        print("\n" + "="*70)
        
        if y is not None:
            return X_scaled, y
        return X_scaled
    


    def handle_imbalance(self, X, y, method='smote'):
        """
        Xử lý mất cân bằng dữ liệu (CHỈ dùng cho training set)
        """
        print("\n" + "="*70)
        print("XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU")
        print("="*70)

        print("Phân bố trước:")
        print(y.value_counts())

        # Kiểm tra nếu đã gần cân bằng thì bỏ qua
        ratio = y.value_counts().min() / y.value_counts().max()
        if ratio > 0.8:
            print("\nDữ liệu đã gần cân bằng → bỏ qua SMOTE")
            return X, y

        if method == 'smote':
            sampler = SMOTE(random_state=42)
        else:
            raise ValueError(f"Method '{method}' chưa hỗ trợ")

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        print("\nPhân bố sau:")
        print(pd.Series(y_resampled).value_counts())

        return X_resampled, y_resampled