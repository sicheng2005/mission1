"""
图像聚类实验代码
任务1: 聚类任务 (25%分数)
- 1.0: 问题的形式化描述 (5%)
- 1.1: 如何处理图像特征 (5%)
- 1.2: 选择合适的聚类算法 (10%)
- 1.3: 评估聚类效果 (5%)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg16
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1.0 问题的形式化描述 ====================
"""
问题形式化描述：
给定一个图像数据集 D = {I₁, I₂, ..., Iₙ}，其中 n = 600 张图像，
每张图像 Iᵢ 属于 6 个类别之一：{cable, tile, bottle, pill, leather, transistor}。

聚类任务的目标是：
1. 将图像从像素空间映射到特征空间：f: I → x ∈ Rᵈ
2. 在特征空间中找到 k = 6 个簇，使得：
   - 同一簇内的图像相似度最大
   - 不同簇间的图像相似度最小
3. 学习一个聚类函数：C: x → {1, 2, ..., k}

这是一个无监督学习问题，因为我们不知道真实的类别标签（虽然我们有ground truth用于评估）。
"""

# ==================== 配置参数 ====================
DATA_DIR = "DM_2025_Dataset/Cluster/dataset"
LABELS_FILE = "DM_2025_Dataset/Cluster/cluster_labels.json"
N_CLUSTERS = 6
FEATURE_EXTRACTION_METHOD = "resnet"  # 可选: "resnet", "vgg", "histogram"
CLUSTERING_METHOD = "kmeans"  # 可选: "kmeans", "dbscan", "agglomerative"

# ==================== 1.1 图像特征提取 ====================
class ImageFeatureExtractor:
    """图像特征提取器"""
    
    def __init__(self, method="resnet"):
        self.method = method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self._init_model()
    
    def _init_model(self):
        """初始化特征提取模型"""
        if self.method == "resnet":
            # 使用ResNet50提取特征
            self.model = resnet50(pretrained=True)
            # 移除最后的全连接层，只保留特征提取部分
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(self.device)
            # ResNet的预处理
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        elif self.method == "vgg":
            # 使用VGG16提取特征
            self.model = vgg16(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.features.children()))
            self.model.eval()
            self.model.to(self.device)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # 使用传统方法：颜色直方图
            self.model = None
            self.transform = None
    
    def extract_features(self, image_path):
        """从单张图像提取特征"""
        if self.method in ["resnet", "vgg"]:
            return self._extract_cnn_features(image_path)
        else:
            return self._extract_histogram_features(image_path)
    
    def _extract_cnn_features(self, image_path):
        """使用CNN提取特征"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
                # 展平特征向量
                features = features.flatten()
            
            return features
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return None
    
    def _extract_histogram_features(self, image_path):
        """使用颜色直方图提取特征"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # 计算RGB三个通道的直方图
            hist_r = np.histogram(image_array[:, :, 0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(image_array[:, :, 1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(image_array[:, :, 2], bins=32, range=(0, 256))[0]
            
            # 归一化
            hist_r = hist_r / (hist_r.sum() + 1e-8)
            hist_g = hist_g / (hist_g.sum() + 1e-8)
            hist_b = hist_b / (hist_b.sum() + 1e-8)
            
            features = np.concatenate([hist_r, hist_g, hist_b])
            return features
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return None
    
    def extract_batch_features(self, image_paths):
        """批量提取特征"""
        features_list = []
        valid_paths = []
        
        print(f"正在使用 {self.method} 方法提取特征...")
        for i, path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(image_paths)} 张图像")
            
            features = self.extract_features(path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(path)
        
        print(f"特征提取完成！共提取 {len(features_list)} 张图像的特征")
        return np.array(features_list), valid_paths


# ==================== 1.2 聚类算法 ====================
class ClusteringAlgorithms:
    """聚类算法集合"""
    
    @staticmethod
    def kmeans_clustering(features, n_clusters=6, random_state=42):
        """K-means聚类"""
        print(f"使用K-means进行聚类，簇数: {n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features)
        return labels, kmeans
    
    @staticmethod
    def dbscan_clustering(features, eps=0.5, min_samples=5):
        """DBSCAN聚类"""
        print(f"使用DBSCAN进行聚类，eps={eps}, min_samples={min_samples}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        return labels, dbscan
    
    @staticmethod
    def agglomerative_clustering(features, n_clusters=6):
        """层次聚类"""
        print(f"使用层次聚类，簇数: {n_clusters}")
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(features)
        return labels, agg


# ==================== 1.3 评估指标 ====================
class ClusteringEvaluator:
    """聚类评估器"""
    
    @staticmethod
    def evaluate(y_true, y_pred, features):
        """评估聚类结果"""
        results = {}
        
        # 调整兰德指数 (ARI)
        ari = adjusted_rand_score(y_true, y_pred)
        results['ARI'] = ari
        
        # 标准化互信息 (NMI)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        results['NMI'] = nmi
        
        # 轮廓系数 (Silhouette Score)
        # 注意：如果簇数太多或样本数太少，可能无法计算
        try:
            silhouette = silhouette_score(features, y_pred)
            results['Silhouette'] = silhouette
        except:
            results['Silhouette'] = None
        
        return results
    
    @staticmethod
    def print_results(results):
        """打印评估结果"""
        print("\n" + "="*50)
        print("聚类评估结果")
        print("="*50)
        print(f"调整兰德指数 (ARI): {results['ARI']:.4f}")
        print(f"标准化互信息 (NMI): {results['NMI']:.4f}")
        if results['Silhouette'] is not None:
            print(f"轮廓系数 (Silhouette): {results['Silhouette']:.4f}")
        print("="*50)


# ==================== 可视化函数 ====================
def visualize_clustering(features, labels, true_labels, method_name, save_path=None):
    """可视化聚类结果"""
    # 使用PCA降维到2D
    print("使用PCA降维到2D进行可视化...")
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 真实标签可视化
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=true_labels, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title('真实标签分布', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    axes[0].grid(True, alpha=0.3)
    
    # 聚类结果可视化
    scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=labels, cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title(f'聚类结果 ({method_name})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()


def map_cluster_labels(y_true, y_pred):
    """
    将聚类标签映射到真实标签，使得混淆矩阵更清晰
    使用匈牙利算法找到最优映射
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    
    # 获取所有唯一的标签
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # 确保标签从0开始连续
    # 重新编码真实标签和预测标签
    true_encoder = {label: i for i, label in enumerate(sorted(unique_true))}
    pred_encoder = {label: i for i, label in enumerate(sorted(unique_pred))}
    
    y_true_encoded = np.array([true_encoder[label] for label in y_true])
    y_pred_encoded = np.array([pred_encoder[label] for label in y_pred])
    
    # 构建混淆矩阵 (行=真实标签, 列=预测标签)
    cm = confusion_matrix(y_true_encoded, y_pred_encoded, 
                          labels=np.arange(len(unique_true)))
    
    # 使用匈牙利算法找到最优映射
    # 我们需要最大化匹配，所以使用负的混淆矩阵
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # 创建映射字典：从编码后的预测标签到编码后的真实标签
    # 然后转换回原始标签空间
    label_mapping = {}
    for true_idx, pred_idx in zip(row_ind, col_ind):
        # pred_idx 是编码后的预测标签
        # true_idx 是编码后的真实标签
        # 需要找到对应的原始标签
        original_pred_label = sorted(unique_pred)[pred_idx]
        original_true_label = sorted(unique_true)[true_idx]
        label_mapping[original_pred_label] = original_true_label
    
    # 应用映射：将原始预测标签映射到对应的真实标签
    y_pred_mapped = np.array([label_mapping.get(label, label) for label in y_pred])
    
    return y_pred_mapped, label_mapping

def plot_cluster_statistics(true_labels, pred_labels, class_names, save_path=None):
    """绘制聚类统计信息"""
    # 创建混淆矩阵风格的统计图
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='聚类结果混淆矩阵',
           ylabel='真实标签',
           xlabel='预测标签')
    
    # 在格子中添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"统计图已保存到: {save_path}")
    
    plt.show()


# ==================== 主函数 ====================
def main():
    """主实验流程"""
    print("="*60)
    print("图像聚类实验")
    print("="*60)
    
    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    image_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png')])
    image_paths = [os.path.join(DATA_DIR, f) for f in image_files]
    
    # 加载真实标签
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    
    # 类别名称映射
    class_names = ['cable', 'tile', 'bottle', 'pill', 'leather', 'transistor']
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    # 2. 提取特征
    print("\n[步骤2] 提取图像特征...")
    extractor = ImageFeatureExtractor(method=FEATURE_EXTRACTION_METHOD)
    features, valid_paths = extractor.extract_batch_features(image_paths)
    
    # 获取对应的真实标签
    true_labels = []
    for path in valid_paths:
        filename = os.path.basename(path)
        class_name = labels_dict.get(filename, 'unknown')
        true_labels.append(class_to_id[class_name])
    true_labels = np.array(true_labels)
    
    print(f"特征维度: {features.shape}")
    print(f"类别分布: {np.bincount(true_labels)}")
    
    # 3. 特征标准化
    print("\n[步骤3] 标准化特征...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 4. 执行聚类
    print("\n[步骤4] 执行聚类...")
    clustering = ClusteringAlgorithms()
    
    if CLUSTERING_METHOD == "kmeans":
        pred_labels, model = clustering.kmeans_clustering(
            features_scaled, n_clusters=N_CLUSTERS
        )
    elif CLUSTERING_METHOD == "dbscan":
        pred_labels, model = clustering.dbscan_clustering(
            features_scaled, eps=0.5, min_samples=5
        )
        # DBSCAN可能产生噪声点（标签为-1）
        n_clusters_found = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        print(f"DBSCAN找到 {n_clusters_found} 个簇")
    elif CLUSTERING_METHOD == "agglomerative":
        pred_labels, model = clustering.agglomerative_clustering(
            features_scaled, n_clusters=N_CLUSTERS
        )
    else:
        raise ValueError(f"未知的聚类方法: {CLUSTERING_METHOD}")
    
    # 5. 标签映射（将聚类标签映射到真实标签）
    print("\n[步骤5] 映射聚类标签到真实标签...")
    pred_labels_mapped, label_mapping = map_cluster_labels(true_labels, pred_labels)
    print(f"标签映射关系: {label_mapping}")
    print("说明: 聚类算法产生的簇标签是随机的，已通过最优匹配映射到真实标签")
    
    # 6. 评估结果
    print("\n[步骤6] 评估聚类结果...")
    # 注意：ARI和NMI不依赖标签映射，所以使用原始pred_labels
    evaluator = ClusteringEvaluator()
    results = evaluator.evaluate(true_labels, pred_labels, features_scaled)
    evaluator.print_results(results)
    
    # 计算准确率（使用映射后的标签）
    accuracy = np.mean(pred_labels_mapped == true_labels)
    print(f"\n聚类准确率（映射后）: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 7. 可视化
    print("\n[步骤7] 可视化结果...")
    # 可视化时使用映射后的标签，使结果更清晰
    visualize_clustering(
        features_scaled, pred_labels_mapped, true_labels, 
        CLUSTERING_METHOD.upper(),
        save_path=f"clustering_visualization_{FEATURE_EXTRACTION_METHOD}_{CLUSTERING_METHOD}.png"
    )
    
    plot_cluster_statistics(
        true_labels, pred_labels_mapped, class_names,
        save_path=f"cluster_statistics_{FEATURE_EXTRACTION_METHOD}_{CLUSTERING_METHOD}.png"
    )
    
    # 8. 保存结果
    print("\n[步骤8] 保存结果...")
    results_dict = {
        'feature_method': FEATURE_EXTRACTION_METHOD,
        'clustering_method': CLUSTERING_METHOD,
        'n_clusters': N_CLUSTERS,
        'n_samples': len(features),
        'evaluation': results,
        'accuracy_mapped': float(accuracy),
        'label_mapping': {str(k): int(v) for k, v in label_mapping.items()},
        'predicted_labels_original': pred_labels.tolist(),
        'predicted_labels_mapped': pred_labels_mapped.tolist()
    }
    
    with open(f"clustering_results_{FEATURE_EXTRACTION_METHOD}_{CLUSTERING_METHOD}.json", 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print("\n实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()

