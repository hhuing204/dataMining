import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

class DataVisualizer:
    def __init__(self, save_dir='results/figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_class_distribution(self, y, title='Phân phối biến mục tiêu'):
        """Vẽ phân phối của biến mục tiêu"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        ax = axes[0]
        counts = y.value_counts()
        colors = ['#ff6b6b', '#4ecdc4']
        counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_title(f'{title} - Số lượng', fontweight='bold', fontsize=12)
        ax.set_xlabel('Revenue')
        ax.set_ylabel('Số lượng')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # Pie chart
        ax = axes[1]
        counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                   colors=colors, startangle=90, explode=(0.05, 0))
        ax.set_ylabel('')
        ax.set_title(f'{title} - Tỷ lệ', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_model_comparison(self, results_df):
        """Vẽ biểu đồ so sánh các mô hình"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        metrics = ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC', 'Train Time (s)']
        colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = results_df[metric].values
            models = results_df.index
            
            bars = ax.barh(models, values, color=colors)
            ax.set_xlabel(metric, fontsize=11)
            ax.set_title(f'So sánh {metric}', fontweight='bold', fontsize=12)
            
            # Thêm giá trị
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{val:.4f}', va='center', fontweight='bold', fontsize=9)
        
        plt.suptitle('SO SÁNH CÁC MÔ HÌNH PHÂN LỚP', fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_confusion_matrices(self, models_results, y_test, class_names=['Không mua', 'Mua']):
        """Vẽ confusion matrices cho tất cả models"""
        n_models = len(models_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.ravel()
        
        for idx, (name, results) in enumerate(models_results.items()):
            cm = confusion_matrix(y_test, results['predictions'])
            
            # Vẽ heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=class_names, yticklabels=class_names,
                       annot_kws={'size': 14, 'weight': 'bold'})
            
            axes[idx].set_title(f'{name}\nF1={results["metrics"]["F1-Score"]:.3f}', 
                               fontweight='bold', fontsize=11)
            axes[idx].set_xlabel('Dự đoán', fontsize=10)
            axes[idx].set_ylabel('Thực tế', fontsize=10)
        
        # Ẩn các subplot thừa
        for idx in range(len(models_results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('MA TRẬN NHẦM LẪN (CONFUSION MATRIX)', fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_roc_curves(self, models_results, y_test):
        """Vẽ ROC curves cho tất cả models"""
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_results)))
        
        for idx, (name, results) in enumerate(models_results.items()):
            if results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
                auc_score = results['metrics']['AUC-ROC']
                plt.plot(fpr, tpr, linewidth=2, color=colors[idx],
                        label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ĐƯỜNG CONG ROC', fontweight='bold', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=15):
        """Vẽ feature importance cho tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, top_n))
            
            plt.barh(range(top_n), importances[indices][::-1], color=colors[::-1])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
            plt.xlabel('Mức độ quan trọng', fontsize=12)
            plt.title(f'TOP {top_n} FEATURES QUAN TRỌNG NHẤT - {model_name}', 
                     fontweight='bold', fontsize=14)
            
            # Thêm giá trị
            for i, v in enumerate(importances[indices][::-1]):
                plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/feature_importance_{model_name.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_training_history(self, cv_results):
        """Vẽ lịch sử training (cross-validation scores)"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(cv_results.keys())
        means = [cv_results[m]['mean'] for m in models]
        stds = [cv_results[m]['std'] for m in models]
        
        x_pos = np.arange(len(models))
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.7,
               ecolor='black', capsize=10, color=plt.cm.Set3(x_pos/len(models)))
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Cross-validation F1-Score')
        ax.set_title('KẾT QUẢ CROSS-VALIDATION (5-FOLD)', fontweight='bold', fontsize=14)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Thêm giá trị
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + 0.02, f'{mean:.3f}±{std:.3f}', 
                   ha='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/cv_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def save_all_figures(self, clf, X_train_processed, y_train, y_test):
        """Lưu tất cả figures"""
        print("\n" + "="*50)
        print("ĐANG TẠO CÁC BIỂU ĐỒ...")
        print("="*50)
        
        # 1. Class distribution
        print("- Đang vẽ phân phối lớp...")
        self.plot_class_distribution(y_train, 'Phân phối trên tập train')
        
        # 2. Model comparison
        print("- Đang vẽ so sánh models...")
        results_df = pd.DataFrame({
            name: results['metrics'] for name, results in clf.results.items()
        }).T
        self.plot_model_comparison(results_df)
        
        # 3. Confusion matrices
        print("- Đang vẽ confusion matrices...")
        self.plot_confusion_matrices(clf.results, y_test)
        
        # 4. ROC curves
        print("- Đang vẽ ROC curves...")
        self.plot_roc_curves(clf.results, y_test)
        
        # 5. Feature importance cho từng model
        print("- Đang vẽ feature importance...")
        for name, results in clf.results.items():
            if hasattr(results['model'], 'feature_importances_'):
                self.plot_feature_importance(
                    results['model'], 
                    [f'F{i}' for i in range(X_train_processed.shape[1])],  # Tạm thời
                    name
                )
        
        # 6. CV results
        print("- Đang vẽ CV results...")
        cv_results = {
            name: {
                'mean': results['metrics']['CV Mean F1'],
                'std': results['metrics']['CV Std F1']
            }
            for name, results in clf.results.items()
        }
        self.plot_training_history(cv_results)
        
        print(f"\n✅ Đã lưu tất cả figures vào thư mục: {self.save_dir}/")
        print(f"   Các file đã tạo:")
        for f in os.listdir(self.save_dir):
            print(f"   - {f}")