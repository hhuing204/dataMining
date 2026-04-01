import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

class AssociationRuleMining:
    def __init__(self, df, transaction_col='Transaction', items_col='Item'):
        self.df = df
        self.transaction_col = transaction_col
        self.items_col = items_col
        self.frequent_itemsets = None
        self.rules = None
        
    def prepare_transaction_data(self):
        """Chuẩn bị dữ liệu giao dịch từ DataFrame"""
        # Nhóm các item theo transaction
        transactions = self.df.groupby(self.transaction_col)[self.items_col].apply(list).tolist()
        
        print(f"Số lượng giao dịch: {len(transactions)}")
        print(f"Ví dụ giao dịch đầu tiên: {transactions[0]}")
        
        self.transactions = transactions
        return transactions
    
    def encode_transactions(self, min_freq=0.01):
        """Mã hóa transactions sang one-hot encoding"""
        te = TransactionEncoder()
        te_ary = te.fit(self.transactions).transform(self.transactions)
        self.df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        print(f"Kích thước ma trận one-hot: {self.df_encoded.shape}")
        print(f"Số items: {len(te.columns_)}")
        
        # Lọc bỏ items hiếm gặp
        item_freq = self.df_encoded.mean()
        rare_items = item_freq[item_freq < min_freq].index.tolist()
        self.df_encoded = self.df_encoded.drop(columns=rare_items)
        
        print(f"Số items sau khi lọc (min_freq={min_freq}): {self.df_encoded.shape[1]}")
        
        return self.df_encoded
    
    def mine_frequent_itemsets(self, min_support=0.05, use_fpgrowth=True):
        """Khai phá tập phổ biến"""
        print(f"\n=== KHAI PHÁ TẬP PHỔ BIẾN (min_support={min_support}) ===")
        
        if use_fpgrowth:
            print("Sử dụng FP-Growth algorithm...")
            self.frequent_itemsets = fpgrowth(self.df_encoded, 
                                             min_support=min_support, 
                                             use_colnames=True)
        else:
            print("Sử dụng Apriori algorithm...")
            self.frequent_itemsets = apriori(self.df_encoded, 
                                            min_support=min_support, 
                                            use_colnames=True)
        
        # Thêm độ dài itemset
        self.frequent_itemsets['length'] = self.frequent_itemsets['itemsets'].apply(len)
        
        print(f"Số tập phổ biến tìm được: {len(self.frequent_itemsets)}")
        print("\nPhân bố theo độ dài:")
        print(self.frequent_itemsets['length'].value_counts().sort_index())
        
        return self.frequent_itemsets
    
    def generate_rules(self, min_confidence=0.5, min_lift=1.0):
        """Sinh luật kết hợp từ tập phổ biến"""
        print(f"\n=== SINH LUẬT KẾT HỢP (min_confidence={min_confidence}, min_lift={min_lift}) ===")
        
        self.rules = association_rules(self.frequent_itemsets, 
                                      metric="confidence",
                                      min_threshold=min_confidence)
        
        # Lọc theo lift
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        # Sắp xếp theo lift
        self.rules = self.rules.sort_values('lift', ascending=False)
        
        print(f"Số luật tìm được: {len(self.rules)}")
        
        return self.rules
    
    def analyze_rules(self):
        """Phân tích các luật kết hợp"""
        if self.rules is None or len(self.rules) == 0:
            print("Không có luật nào để phân tích")
            return
        
        print("\n=== PHÂN TÍCH LUẬT KẾT HỢP ===")
        print(f"Tổng số luật: {len(self.rules)}")
        print(f"\nThống kê các chỉ số:")
        print(self.rules[['support', 'confidence', 'lift']].describe())
        
        # Top 10 luật theo lift
        print("\nTop 10 luật có lift cao nhất:")
        top_rules = self.rules.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        for idx, row in top_rules.iterrows():
            print(f"{set(row['antecedents'])} -> {set(row['consequents'])}: "
                  f"sup={row['support']:.3f}, conf={row['confidence']:.3f}, lift={row['lift']:.3f}")
    
    def visualize_rules(self):
        """Visualize các luật kết hợp"""
        if self.rules is None or len(self.rules) == 0:
            print("Không có luật để visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Scatter plot: support vs confidence
        axes[0,0].scatter(self.rules['support'], self.rules['confidence'], 
                         c=self.rules['lift'], cmap='viridis', alpha=0.6, s=50)
        axes[0,0].set_xlabel('Support')
        axes[0,0].set_ylabel('Confidence')
        axes[0,0].set_title('Support vs Confidence (color = Lift)', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(axes[0,0].collections[0], ax=axes[0,0])
        
        # Histogram of lift
        axes[0,1].hist(self.rules['lift'], bins=20, color='steelblue', edgecolor='black')
        axes[0,1].set_xlabel('Lift')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Lift Values', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Heatmap of top rules
        if len(self.rules) > 20:
            plot_rules = self.rules.head(20)
        else:
            plot_rules = self.rules
            
        # Parallel coordinates plot không phù hợp, dùng bar plot thay thế
        rule_names = [f"{list(a)[0] if len(a)==1 else '...'}->{list(c)[0]}" 
                     for a, c in zip(plot_rules['antecedents'], plot_rules['consequents'])]
        
        axes[1,0].barh(range(len(plot_rules)), plot_rules['lift'].values)
        axes[1,0].set_yticks(range(len(plot_rules)))
        axes[1,0].set_yticklabels(rule_names, fontsize=8)
        axes[1,0].set_xlabel('Lift')
        axes[1,0].set_title('Top Rules by Lift', fontweight='bold')
        axes[1,0].invert_yaxis()
        axes[1,0].grid(True, alpha=0.3)
        
        # Scatter plot: support vs lift
        axes[1,1].scatter(self.rules['support'], self.rules['lift'], 
                         c=self.rules['confidence'], cmap='plasma', alpha=0.6, s=50)
        axes[1,1].set_xlabel('Support')
        axes[1,1].set_ylabel('Lift')
        axes[1,1].set_title('Support vs Lift (color = Confidence)', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('../results/figures/association_rules.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def find_rules_for_item(self, item, as_consequent=False):
        """Tìm các luật liên quan đến một item cụ thể"""
        if self.rules is None:
            print("Chưa có luật nào")
            return None
        
        if as_consequent:
            mask = self.rules['consequents'].apply(lambda x: item in x)
            title = f"Rules with '{item}' as consequent"
        else:
            mask = self.rules['antecedents'].apply(lambda x: item in x)
            title = f"Rules with '{item}' as antecedent"
        
        item_rules = self.rules[mask].sort_values('lift', ascending=False)
        
        print(f"\n=== {title} ===")
        print(f"Tìm thấy {len(item_rules)} luật")
        
        if len(item_rules) > 0:
            for idx, row in item_rules.head(10).iterrows():
                print(f"{set(row['antecedents'])} -> {set(row['consequents'])}: "
                      f"lift={row['lift']:.3f}, conf={row['confidence']:.3f}")
        
        return item_rules
    
    def plot_item_network(self, top_n=20):
        """Vẽ mạng lưới items (simplified version)"""
        if self.rules is None or len(self.rules) == 0:
            return
        
        # Lấy top rules
        top_rules = self.rules.nlargest(top_n, 'lift')
        
        # Tạo adjacency matrix đơn giản
        items = set()
        for _, rule in top_rules.iterrows():
            items.update(rule['antecedents'])
            items.update(rule['consequents'])
        
        items = list(items)
        n_items = len(items)
        
        if n_items == 0:
            return
        
        # Tạo ma trận kề đơn giản
        adj_matrix = pd.DataFrame(0, index=items, columns=items)
        
        for _, rule in top_rules.iterrows():
            for a in rule['antecedents']:
                for c in rule['consequents']:
                    adj_matrix.loc[a, c] = rule['lift']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(adj_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Lift'})
        plt.title(f'Item Association Network (Top {top_n} Rules)', fontweight='bold')
        plt.tight_layout()
        plt.savefig('../results/figures/item_network.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def parameter_sensitivity(self):
        """Phân tích độ nhạy của tham số"""
        print("\n=== PHÂN TÍCH ĐỘ NHẠY THAM SỐ ===")
        
        support_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]
        results = []
        
        for min_sup in support_thresholds:
            try:
                # Khai phá tập phổ biến
                itemsets = apriori(self.df_encoded, min_support=min_sup, use_colnames=True)
                
                # Sinh luật
                if len(itemsets) > 0:
                    rules = association_rules(itemsets, metric="confidence", min_threshold=0.5)
                    
                    results.append({
                        'min_support': min_sup,
                        'n_itemsets': len(itemsets),
                        'n_rules': len(rules),
                        'avg_lift': rules['lift'].mean() if len(rules) > 0 else 0
                    })
            except:
                results.append({
                    'min_support': min_sup,
                    'n_itemsets': 0,
                    'n_rules': 0,
                    'avg_lift': 0
                })
        
        results_df = pd.DataFrame(results)
        print(results_df)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(results_df['min_support'], results_df['n_itemsets'], 'bo-')
        axes[0].set_xlabel('Min Support')
        axes[0].set_ylabel('Number of Itemsets')
        axes[0].set_title('Itemsets vs Min Support')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(results_df['min_support'], results_df['n_rules'], 'ro-')
        axes[1].set_xlabel('Min Support')
        axes[1].set_ylabel('Number of Rules')
        axes[1].set_title('Rules vs Min Support')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(results_df['min_support'], results_df['avg_lift'], 'go-')
        axes[2].set_xlabel('Min Support')
        axes[2].set_ylabel('Average Lift')
        axes[2].set_title('Avg Lift vs Min Support')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/figures/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df

# Sử dụng
if __name__ == "__main__":
    # Tạo dữ liệu mẫu cho association rule mining
    # Giả sử chúng ta có dữ liệu giỏ hàng
    np.random.seed(42)
    
    # Tạo transactions mẫu
    items = ['Bread', 'Milk', 'Butter', 'Eggs', 'Cheese', 'Yogurt', 'Cereal', 'Juice']
    n_transactions = 1000
    
    transactions = []
    for _ in range(n_transactions):
        # Random số lượng items trong transaction
        n_items = np.random.randint(1, 6)
        transaction = np.random.choice(items, size=n_items, replace=False)
        transactions.append(list(transaction))
    
    # Tạo DataFrame
    df_trans = pd.DataFrame({
        'Transaction': [f'T{i+1}' for i in range(n_transactions)],
        'Items': [','.join(t) for t in transactions]
    })
    
    # Explode items
    df_expanded = df_trans.assign(Item=df_trans['Items'].str.split(',')).explode('Item')
    
    # Association Rule Mining
    arm = AssociationRuleMining(df_expanded, transaction_col='Transaction', items_col='Item')
    
    # Chuẩn bị dữ liệu
    transactions = arm.prepare_transaction_data()
    
    # Encode
    df_encoded = arm.encode_transactions(min_freq=0.05)
    
    # Khai phá tập phổ biến
    frequent_itemsets = arm.mine_frequent_itemsets(min_support=0.05, use_fpgrowth=True)
    
    # Sinh luật
    rules = arm.generate_rules(min_confidence=0.5, min_lift=1.0)
    
    # Phân tích
    arm.analyze_rules()
    arm.visualize_rules()
    
    # Tìm luật cho item cụ thể
    arm.find_rules_for_item('Milk', as_consequent=False)
    
    # Phân tích độ nhạy
    arm.parameter_sensitivity()