import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

class ClusteringAnalysis:
    def __init__(self, X, feature_names=None):
        self.X = X.copy()
        self.feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(X.shape[1])]
        self.X_scaled = StandardScaler().fit_transform(self.X)
        
    def find_optimal_k(self, k_range=range(2, 11)):
        """Tìm số cluster tối ưu cho K-means"""
        print("\n=== TÌM SỐ CLUSTER TỐI ƯU ===")
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, kmeans.labels_))
            
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")
        
        # Vẽ biểu đồ Elbow
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow method
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia', fontsize=12)
        axes[0].set_title('Elbow Method for Optimal k', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score for Optimal k', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/figures/optimal_k.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Chọn k tối ưu
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\nK optimal: {optimal_k}")
        
        return optimal_k
    
    def kmeans_clustering(self, n_clusters=3):
        """K-means clustering"""
        print(f"\n=== K-MEANS CLUSTERING (k={n_clusters}) ===")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.X_scaled)
        
        # Đánh giá
        silhouette = silhouette_score(self.X_scaled, labels)
        calinski = calinski_harabasz_score(self.X_scaled, labels)
        davies = davies_bouldin_score(self.X_scaled, labels)
        
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski:.4f}")
        print(f"Davies-Bouldin Score: {davies:.4f}")
        
        return kmeans, labels
    
    def hierarchical_clustering(self, n_clusters=3):
        """Hierarchical clustering"""
        print(f"\n=== HIERARCHICAL CLUSTERING (k={n_clusters}) ===")
        
        # Tính linkage matrix
        linkage_matrix = linkage(self.X_scaled, method='ward')
        
        # Vẽ dendrogram
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, truncate_mode='lastp', p=30, 
                  leaf_rotation=90., leaf_font_size=8.)
        plt.title('Hierarchical Clustering Dendrogram', fontweight='bold')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.axhline(y=linkage_matrix[-n_clusters+1, 2], color='r', linestyle='--', 
                   label=f'Cut at {n_clusters} clusters')
        plt.legend()
        plt.tight_layout()
        plt.savefig('../results/figures/dendrogram.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Fit model
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(self.X_scaled)
        
        # Đánh giá
        silhouette = silhouette_score(self.X_scaled, labels)
        print(f"Silhouette Score: {silhouette:.4f}")
        
        return hierarchical, labels
    
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """DBSCAN clustering"""
        print(f"\n=== DBSCAN CLUSTERING (eps={eps}, min_samples={min_samples}) ===")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
        
        if n_clusters > 1:
            silhouette = silhouette_score(self.X_scaled[labels != -1], labels[labels != -1])
            print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
        
        return dbscan, labels
    
    def visualize_clusters(self, labels, method_name):
        """Visualize clusters using PCA"""
        # Giảm chiều với PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        plt.figure(figsize=(10, 8))
        
        # Màu sắc cho clusters
        unique_labels = set(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Noise points
                col = 'black'
                marker = 'x'
                label = 'Noise'
            else:
                marker = 'o'
                label = f'Cluster {k}'
            
            class_member_mask = (labels == k)
            xy = X_pca[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                       s=50, edgecolor='k', label=label)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title(f'Clustering Results - {method_name}', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../results/figures/clusters_{method_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return X_pca
    
    def analyze_clusters(self, labels, original_data):
        """Phân tích đặc điểm của từng cluster"""
        print("\n=== CLUSTER ANALYSIS ===")
        
        # Thêm cluster labels vào data
        data_with_cluster = original_data.copy()
        data_with_cluster['Cluster'] = labels
        
        # Thống kê theo cluster
        cluster_stats = []
        for cluster in sorted(set(labels)):
            if cluster == -1:
                continue  # Bỏ qua noise
                
            cluster_data = data_with_cluster[data_with_cluster['Cluster'] == cluster]
            
            stats = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Percentage': len(cluster_data)/len(data_with_cluster)*100,
                'Revenue_Rate': cluster_data['Revenue'].mean() if 'Revenue' in cluster_data else None
            }
            
            # Thêm mean của các numeric features
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Cluster':
                    stats[f'{col}_mean'] = cluster_data[col].mean()
            
            cluster_stats.append(stats)
        
        stats_df = pd.DataFrame(cluster_stats)
        print(stats_df.round(2))
        
        # Visualize cluster characteristics
        if 'Revenue' in data_with_cluster.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Revenue rate by cluster
            cluster_revenue = data_with_cluster.groupby('Cluster')['Revenue'].mean() * 100
            axes[0].bar(cluster_revenue.index, cluster_revenue.values, color='steelblue')
            axes[0].set_xlabel('Cluster')
            axes[0].set_ylabel('Revenue Rate (%)')
            axes[0].set_title('Tỷ lệ mua hàng theo Cluster', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Cluster size
            cluster_size = data_with_cluster['Cluster'].value_counts()
            axes[1].pie(cluster_size.values, labels=[f'Cluster {i}' for i in cluster_size.index],
                       autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Phân bố kích thước Cluster', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('../results/figures/cluster_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return stats_df
    
    def compare_clustering_methods(self):
        """So sánh các phương pháp clustering"""
        print("\n=== COMPARING CLUSTERING METHODS ===")
        
        methods = {
            'K-means (k=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
            'K-means (k=4)': KMeans(n_clusters=4, random_state=42, n_init=10),
            'K-means (k=5)': KMeans(n_clusters=5, random_state=42, n_init=10),
            'Hierarchical (k=3)': AgglomerativeClustering(n_clusters=3),
            'DBSCAN (eps=0.5)': DBSCAN(eps=0.5, min_samples=5),
            'DBSCAN (eps=0.7)': DBSCAN(eps=0.7, min_samples=5)
        }
        
        results = []
        
        for name, method in methods.items():
            labels = method.fit_predict(self.X_scaled)
            
            # Tính metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            metrics = {
                'Method': name,
                'N_Clusters': n_clusters,
                'N_Noise': n_noise,
                'Noise_%': n_noise/len(labels)*100
            }
            
            if n_clusters > 1:
                # Silhouette score (bỏ qua noise nếu có)
                if -1 in labels:
                    mask = labels != -1
                    metrics['Silhouette'] = silhouette_score(self.X_scaled[mask], labels[mask])
                else:
                    metrics['Silhouette'] = silhouette_score(self.X_scaled, labels)
                
                metrics['Calinski'] = calinski_harabasz_score(self.X_scaled, labels)
                metrics['Davies'] = davies_bouldin_score(self.X_scaled, labels)
            
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        print(results_df.round(4))
        
        return results_df

# Sử dụng
if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessing import DataPreprocessor
    
    # Load và preprocess data
    loader = DataLoader()
    df = loader.load_data()
    
    preprocessor = DataPreprocessor(df)
    preprocessor.run_pipeline()
    X, y = preprocessor.get_preprocessed_data()
    
    # Chỉ lấy numeric features cho clustering
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]
    
    # Clustering
    clustering = ClusteringAnalysis(X_numeric)
    
    # Tìm k optimal
    optimal_k = clustering.find_optimal_k()
    
    # K-means
    kmeans, labels_kmeans = clustering.kmeans_clustering(n_clusters=optimal_k)
    clustering.visualize_clusters(labels_kmeans, 'KMeans')
    
    # Hierarchical
    hierarchical, labels_hier = clustering.hierarchical_clustering(n_clusters=optimal_k)
    clustering.visualize_clusters(labels_hier, 'Hierarchical')
    
    # DBSCAN
    dbscan, labels_dbscan = clustering.dbscan_clustering(eps=0.5, min_samples=5)
    clustering.visualize_clusters(labels_dbscan, 'DBSCAN')
    
    # Phân tích clusters
    clustering.analyze_clusters(labels_kmeans, df)
    
    # So sánh các methods
    clustering.compare_clustering_methods()