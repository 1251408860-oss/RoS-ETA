import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os
import warnings
import copy

# ==========================================
# 1. 全局配置与环境准备
# ==========================================
warnings.filterwarnings('ignore')

# 这里的路径是根据你截图中的目录结构配置的
# 假设脚本运行在 Weakpas_expre 根目录下或者与这些文件夹同级
DATASETS_CONFIG = {
    "CIC-IDS2017": {
        "path": "./CIC-IDS2017_data/CIC-IDS2017_Processed_Lite.csv",
        "poison_ratio": 0.10
    },
    "UNSW-NB15": {
        "path": "./UNSW-NB15_data/UNSW-NB15_Processed_Lite.csv",
        "poison_ratio": 0.10
    }
}

RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=== RoS-ETA Systematically Refined Code (Multi-Dataset Support) ===")


# ==========================================
# 2. 核心类定义 (保持原样，核心逻辑不变)
# ==========================================

class ThreatModelLoader:
    """负责数据的加载、清洗与投毒模拟"""

    def __init__(self, filepath, poison_ratio=0.10):
        self.filepath = filepath
        self.poison_ratio = poison_ratio

    def load_and_poison(self):
        if not os.path.exists(self.filepath):
            print(f"   [Error] File not found: {self.filepath}")
            print("   Generating dummy data for demonstration to prevent crash...")
            return self._generate_dummy_data()

        # 优化读取：只读取需要的列，减少内存消耗
        try:
            df = pd.read_csv(self.filepath)
        except Exception as e:
            print(f"   [Error] Read CSV failed: {e}")
            return self._generate_dummy_data()

        # 标签标准化：确保 Is_Attack 列存在
        if 'Is_Attack' not in df.columns:
            if 'Label' in df.columns:
                # 尝试根据 Label 内容推断 (CIC 风格)
                df['Is_Attack'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
            else:
                print("   [Warning] No Label/Is_Attack found. Assuming all benign.")
                df['Is_Attack'] = 0

        # 确保是 int 类型
        df['Is_Attack'] = df['Is_Attack'].fillna(0).astype(int)

        df_benign_pool = df[df['Is_Attack'] == 0]

        # 攻击池筛选：优先选择 DoS 攻击作为投毒数据
        # UNSW-NB15 的 Label 列比较杂，attack_cat 列包含具体类别
        df_attack_pool = pd.DataFrame()

        # 尝试寻找 DoS 相关的攻击
        potential_dos_mask = pd.Series([False] * len(df))

        if 'Label' in df.columns:  # CIC-IDS2017 风格
            potential_dos_mask |= df['Label'].astype(str).str.contains('DoS', case=False, na=False)

        if 'attack_cat' in df.columns:  # UNSW-NB15 风格
            potential_dos_mask |= df['attack_cat'].astype(str).str.contains('DoS', case=False, na=False)

        if potential_dos_mask.sum() > 100:
            df_attack_pool = df[potential_dos_mask]
        else:
            # 如果没找到明确的 DoS 标签，使用所有攻击数据
            df_attack_pool = df[df['Is_Attack'] == 1]

        # 采样配置
        n_experiment_total = 20000
        if len(df) < n_experiment_total:
            n_experiment_total = len(df)

        n_poison = int(n_experiment_total * self.poison_ratio)
        n_clean = n_experiment_total - n_poison

        def robust_sample(pool, n):
            if len(pool) == 0: return pd.DataFrame()
            replace = len(pool) < n
            return pool.sample(n=n, replace=replace, random_state=42)

        train_clean = robust_sample(df_benign_pool, n_clean)
        train_poison = robust_sample(df_attack_pool, n_poison)

        if len(train_clean) == 0 or len(train_poison) == 0:
            print("   [Warning] Insufficient data for sampling. Returning raw sample.")
            return df.sample(n=min(2000, len(df)), random_state=42)

        # 混合并打乱
        df_dirty = pd.concat([train_clean, train_poison]).sample(frac=1, random_state=42).reset_index(drop=True)
        return df_dirty

    def _generate_dummy_data(self):
        N = 5000
        time_clean = np.random.exponential(scale=2.0, size=int(N * 0.9))
        vol_clean = time_clean * 100 + np.random.normal(0, 10, int(N * 0.9))
        time_attack = np.random.uniform(0.1, 5.0, int(N * 0.1))
        vol_attack = time_attack * 100 + np.random.normal(0, 300, int(N * 0.1))

        df = pd.DataFrame({
            'log_Flow Duration': np.concatenate([time_clean, time_attack]),
            'log_Total Length of Fwd Packets': np.concatenate([vol_clean, vol_attack]),
            'Is_Attack': np.concatenate([np.zeros(int(N * 0.9)), np.ones(int(N * 0.1))]),
            'Label': ['Dummy'] * N
        })
        return df.sample(frac=1, random_state=42).reset_index(drop=True)


class FeatureEngineer:
    """
    集成了 NeurIPS 论文理论：
    1. 对数空间映射 (Physical Linearization)
    2. 显式高阶特征嵌入 (Explicit High-Order Features)
    """

    def __init__(self, target_col):
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.tau = None
        self.is_fitted = False

    def fit(self, df):
        """只在训练集上调用，计算统计量"""
        X = self._extract_raw_features(df)
        self.tau = X.quantile(0.99).fillna(0)
        X_clipped = X.clip(upper=self.tau, axis=1)
        self.scaler.fit(X_clipped)
        self.is_fitted = True
        return self

    def transform(self, df):
        """应用训练集的统计量到任何数据集"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted on training data first!")

        X = self._extract_raw_features(df)
        X_clipped = X.clip(upper=self.tau, axis=1)
        X_scaled = self.scaler.transform(X_clipped)

        if self.target_col in df.columns:
            y_target = df[self.target_col].values
        else:
            real_col = self.target_col.replace('log_', '')
            if real_col in df.columns:
                y_target = np.log1p(df[real_col].values)
            else:
                y_target = np.zeros(len(df))

        return np.nan_to_num(X_scaled), np.nan_to_num(y_target)

    def _extract_raw_features(self, df):
        """特征构造核心逻辑"""
        X = df.copy()

        base_feature = None
        if self.target_col not in X.columns:
            real_col = self.target_col.replace('log_', '')
            if real_col in X.columns:
                base_feature = np.log1p(X[real_col].values)
                X[self.target_col] = base_feature
            else:
                base_feature = np.zeros(len(X))
                X[self.target_col] = base_feature
        else:
            base_feature = X[self.target_col].values

        # 显式高阶矩特征
        X[f'{self.target_col}_sq'] = base_feature ** 2
        X[f'{self.target_col}_cub'] = base_feature ** 3
        X[f'{self.target_col}_quart'] = base_feature ** 4

        # 剔除掉非数值和不相关列
        cols_to_drop = [self.target_col, 'Is_Attack', 'Label', 'attack_cat',
                        'Destination Port', 'Source Port', 'Protocol', 'Timestamp',
                        'Flow ID', 'Source IP', 'Destination IP', 'srcip', 'sport',
                        'dstip', 'dsport', 'proto', 'state', 'service']

        X_feat = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
        # 只保留数值型特征，自动过滤掉 UNSW 的字符串列
        X_feat = X_feat.select_dtypes(include=[np.number]).fillna(0)

        return X_feat


class SpectralSanitizer:
    """基于谱分析的无监督清洗"""

    def __init__(self, max_iter=20):
        self.max_iter = max_iter

    def get_weights(self, X):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.max_iter):
            try:
                Cov = (X.T * w) @ X
                eigvals, eigvecs = eigh(Cov, subset_by_index=[n_features - 1, n_features - 1])
                v = eigvecs[:, 0]
                scores = (X @ v) ** 2
                max_score = np.max(scores)
                if max_score < 1e-9: break
                decay = scores / max_score
                w_new = w * (1 - decay)
                if np.sum(w_new) == 0: break
                w_new /= np.sum(w_new)
                if np.linalg.norm(w_new - w) < 1e-4: break
                w = w_new
            except:
                break

        w_reshaped = w.reshape(-1, 1)
        # 只有当样本量足够时才做KMeans
        if len(w_reshaped) > 10:
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(w_reshaped)
                threshold = np.mean(kmeans.cluster_centers_)
                clean_weights = np.where(w < threshold, 0, w)
            except:
                clean_weights = w  # Fallback
        else:
            clean_weights = w

        if np.sum(clean_weights) > 0:
            clean_weights /= np.sum(clean_weights)

        return clean_weights


class RobustDetector:
    """Huber Loss 回归器"""

    def __init__(self, alpha=1e-5):
        self.model = SGDRegressor(
            loss='huber', penalty='l1', alpha=alpha, epsilon=1.35,
            max_iter=3000, tol=1e-4, learning_rate='adaptive', eta0=0.01, random_state=42
        )

    def fit(self, X, y, sample_weights):
        mask = sample_weights > 0
        if np.sum(mask) > 10:
            self.model.fit(X[mask], y[mask], sample_weight=sample_weights[mask])
        else:
            self.model.fit(X, y)

    def predict_score(self, X, y):
        return np.abs(y - self.model.predict(X))


# ==========================================
# 3. 主程序逻辑 (循环处理所有数据集)
# ==========================================

if __name__ == "__main__":

    for dataset_name, config in DATASETS_CONFIG.items():
        print("\n" + "#" * 60)
        print(f" STARTING EXPERIMENT FOR DATASET: {dataset_name}")
        print("#" * 60)

        input_file = config['path']
        poison_ratio = config['poison_ratio']

        # --- [Step 1] 数据加载与初始切分 ---
        print(f"[1] Loading Data from {input_file}...")
        loader = ThreatModelLoader(input_file, poison_ratio=poison_ratio)
        df_all = loader.load_and_poison()

        if df_all.empty:
            print(f"   [Skipping] Dataframe is empty for {dataset_name}.")
            continue

        # 三方切分：Train (60%), Validation (20%), Test (20%)
        try:
            df_train_full, df_test = train_test_split(df_all, test_size=0.2, stratify=df_all['Is_Attack'],
                                                      random_state=42)
            df_train, df_val = train_test_split(df_train_full, test_size=0.25, stratify=df_train_full['Is_Attack'],
                                                random_state=42)
        except ValueError as e:
            print(f"   [Error] Split failed (likely insufficient classes): {e}")
            continue

        print(f"   Train Set: {len(df_train)}")
        print(f"   Val Set:   {len(df_val)}")
        print(f"   Test Set:  {len(df_test)}")

        VIEWS = {
            'Time': 'log_Flow Duration',
            'Volume': 'log_Total Length of Fwd Packets'
        }

        # 检查必要的列是否存在
        missing_cols = [col for col in VIEWS.values() if col not in df_all.columns]
        if missing_cols:
            print(f"   [Error] Missing columns in {dataset_name}: {missing_cols}")
            print("   Please check your preprocessing script.")
            continue

        scores = {
            'val': {'y_true': df_val['Is_Attack'].values},
            'test': {'y_true': df_test['Is_Attack'].values}
        }

        # --- [Step 2] 逐个视图训练 ---
        print("\n[2] Training Robust Models per View...")

        detectors = {}
        engineers = {}

        # 存储 Test Set 的原始残差用于最后画图
        raw_test_resid = {}

        for v_name, target in VIEWS.items():
            print(f"   > View: {v_name}")

            # 2.1 特征工程
            eng = FeatureEngineer(target)
            eng.fit(df_train)
            engineers[v_name] = eng

            X_train, y_train = eng.transform(df_train)
            X_val, y_val = eng.transform(df_val)
            X_test, y_test = eng.transform(df_test)

            # 2.2 谱过滤
            sanitizer = SpectralSanitizer()
            train_weights = sanitizer.get_weights(X_train)

            # 2.3 训练回归器
            detector = RobustDetector()
            detector.fit(X_train, y_train, train_weights)
            detectors[v_name] = detector

            # 2.4 计算残差
            train_resid = detector.predict_score(X_train, y_train)
            val_resid = detector.predict_score(X_val, y_val)
            test_resid = detector.predict_score(X_test, y_test)

            raw_test_resid[v_name] = test_resid

            resid_scaler = StandardScaler()
            resid_scaler.fit(train_resid.reshape(-1, 1))

            scores['val'][v_name] = resid_scaler.transform(val_resid.reshape(-1, 1)).ravel()
            scores['test'][v_name] = resid_scaler.transform(test_resid.reshape(-1, 1)).ravel()

            # 安全处理 AUC 计算（防止全是单一类别报错）
            try:
                auc_val = roc_auc_score(scores['val']['y_true'], scores['val'][v_name])
                print(f"     Val AUC ({v_name}): {auc_val:.4f}")
            except:
                print(f"     Val AUC ({v_name}): N/A (One class only)")

        # --- [Step 3] 融合参数寻优 ---
        print("\n[3] Optimizing Dual-View Fusion...")

        w_range = np.linspace(0, 1, 21)
        best_auc = 0
        best_w = 0.5

        val_time = scores['val']['Time']
        val_vol = scores['val']['Volume']
        y_val_true = scores['val']['y_true']

        for w in w_range:
            fused = w * val_time + (1 - w) * val_vol
            try:
                auc = roc_auc_score(y_val_true, fused)
                if auc > best_auc:
                    best_auc = auc
                    best_w = w
            except:
                pass

        print(f"   Best Weight: w_Time={best_w:.2f}, w_Vol={1 - best_w:.2f}")

        # --- [Step 4] 最终评估 ---
        print("\n[4] Final Evaluation...")

        test_time = scores['test']['Time']
        test_vol = scores['test']['Volume']
        y_test_true = scores['test']['y_true']

        final_scores = best_w * test_time + (1 - best_w) * test_vol

        try:
            final_auc = roc_auc_score(y_test_true, final_scores)
            print("=" * 40)
            print(f"[{dataset_name}] ROBUSTNESS RESULT (Test Set): AUC = {final_auc:.4f}")
            print("=" * 40)
        except:
            print(f"[{dataset_name}] Evaluation Failed (Check Data Distribution)")
            final_auc = 0

        # --- [Step 5] 绘制并保存决策边界图 ---
        plt.figure(figsize=(8, 8))
        idx = np.random.choice(len(y_test_true), size=min(1000, len(y_test_true)), replace=False)

        # 散点图
        plt.scatter(test_time[idx][y_test_true[idx] == 0], test_vol[idx][y_test_true[idx] == 0],
                    c='green', alpha=0.5, label='Benign', s=20)
        plt.scatter(test_time[idx][y_test_true[idx] == 1], test_vol[idx][y_test_true[idx] == 1],
                    c='red', alpha=0.6, label='Poison Attack', s=20, marker='x')

        plt.xlabel('Standardized Residual (Time View)')
        plt.ylabel('Standardized Residual (Volume View)')
        plt.title(f'{dataset_name}: Dual-View Decision Boundary (AUC={final_auc:.3f})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)

        # 保存图片，文件名带有数据集名称，防止覆盖
        save_path = os.path.join(RESULTS_DIR, f'RoS_ETA_Boundary_{dataset_name}.png')
        plt.savefig(save_path)
        print(f"Boundary plot saved to: {save_path}")
        plt.close()  # 关闭画布，准备下一轮循环

    print("\nAll experiments completed.")