import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RANSACRegressor
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

# ==========================================
# 1. Global Configuration
# ==========================================
DATASETS_CONFIG = {
    "CIC-IDS2017": {
        "path": "./CIC-IDS2017_data/CIC-IDS2017_Processed_Lite.csv",
        "poison_ratio": 0.10
    },
    "UNSW-NB15": {
        "path": "./UNSW-NB15_data/UNSW-NB15_Processed_Lite.csv",
        "poison_ratio": 0.10
    },
    "CIC-IoT2023": {
        "path": "./CICIOT23_data/CICIOT23_Processed_Lite.csv",
        "poison_ratio": 0.10
    }
}

RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=== RoS-ETA Systematically Refined Code (Multi-Dataset Support) ===")

# ==========================================
# 2. Core Classes
# ==========================================

class ThreatModelLoader:
    def __init__(self, filepath, poison_ratio=0.10):
        self.filepath = filepath
        self.poison_ratio = poison_ratio

    def load_and_poison(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Data file missing: {self.filepath}")

        df = pd.read_csv(self.filepath)

        if 'Is_Attack' not in df.columns:
            if 'Label' in df.columns:
                df['Is_Attack'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
            else:
                df['Is_Attack'] = 0

        df['Is_Attack'] = df['Is_Attack'].fillna(0).astype(int)

        df_benign_pool = df[df['Is_Attack'] == 0]
        df_attack_pool = pd.DataFrame()
        potential_dos_mask = pd.Series([False] * len(df))

        if 'Label' in df.columns:
            potential_dos_mask |= df['Label'].astype(str).str.contains('DoS', case=False, na=False)
        if 'attack_cat' in df.columns:
            potential_dos_mask |= df['attack_cat'].astype(str).str.contains('DoS', case=False, na=False)

        if potential_dos_mask.sum() > 100:
            df_attack_pool = df[potential_dos_mask]
        else:
            df_attack_pool = df[df['Is_Attack'] == 1]

        n_experiment_total = min(20000, len(df))
        n_poison = int(n_experiment_total * self.poison_ratio)
        n_clean = n_experiment_total - n_poison

        def robust_sample(pool, n):
            if len(pool) == 0: return pd.DataFrame()
            return pool.sample(n=n, replace=(len(pool) < n), random_state=42)

        train_clean = robust_sample(df_benign_pool, n_clean)
        train_poison = robust_sample(df_attack_pool, n_poison)

        if len(train_clean) == 0 or len(train_poison) == 0:
            return df.sample(n=min(2000, len(df)), random_state=42)

        df_dirty = pd.concat([train_clean, train_poison]).sample(frac=1, random_state=42).reset_index(drop=True)
        return df_dirty


class FeatureEngineer:
    def __init__(self, target_col):
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.tau = None
        self.is_fitted = False

    def fit(self, df):
        X = self._extract_raw_features(df)
        self.tau = X.quantile(0.99).fillna(0)
        X_clipped = X.clip(upper=self.tau, axis=1)
        self.scaler.fit(X_clipped)
        self.is_fitted = True
        return self

    def transform(self, df):
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
        X = df.copy()

        if self.target_col not in X.columns:
            real_col = self.target_col.replace('log_', '')
            base_feature = np.log1p(X[real_col].values) if real_col in X.columns else np.zeros(len(X))
            X[self.target_col] = base_feature
        else:
            base_feature = X[self.target_col].values

        entropy_val = None
        std_candidates = [
            'Fwd Packet Length Std', 'Packet Length Std', 'Fwd Pkt Len Std',
            'Bwd Packet Length Std', 'Flow IAT Std', 'Fwd IAT Std',
            'sload', 'Sload', 'std', 'std_dev',
            'log_Fwd Packet Length Std', 'log_Packet Length Std'
        ]

        for cand in std_candidates:
            if cand in X.columns:
                val = X[cand].values
                entropy_val = val if cand.startswith('log_') else np.log1p(val)
                break

        if entropy_val is None and 'Total Length of Fwd Packets' in X.columns and 'Total Fwd Packets' in X.columns:
            avg_len = X['Total Length of Fwd Packets'] / (X['Total Fwd Packets'] + 1e-6)
            entropy_val = np.log1p(avg_len.values)

        if entropy_val is None:
            vol_col, cnt_col = 'log_Total Length of Fwd Packets', 'log_Total Fwd Packets'
            if vol_col in X.columns and cnt_col in X.columns:
                entropy_val = X[vol_col].values - X[cnt_col].values

        if entropy_val is not None:
            entropy_val = np.nan_to_num(entropy_val)
            X['phys_entropy'] = entropy_val
            X['interaction_base_entropy'] = base_feature * entropy_val
        else:
            X['phys_entropy'] = np.zeros(len(X))
            X['interaction_base_entropy'] = np.zeros(len(X))

        X[f'{self.target_col}_sq'] = base_feature ** 2
        X[f'{self.target_col}_cub'] = base_feature ** 3

        cols_to_drop = [self.target_col, 'Is_Attack', 'Label', 'attack_cat',
                        'Destination Port', 'Source Port', 'Protocol', 'Timestamp',
                        'Flow ID', 'Source IP', 'Destination IP', 'srcip', 'sport',
                        'dstip', 'dsport', 'proto', 'state', 'service',
                        'Fwd Packet Length Std', 'Total Length of Fwd Packets', 'Total Fwd Packets']

        X_feat = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')
        return X_feat.select_dtypes(include=[np.number]).fillna(0)


class SpectralSanitizer:
    def __init__(self, max_iter=20):
        self.max_iter = max_iter

    def get_weights(self, X):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.max_iter):
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

        w_reshaped = w.reshape(-1, 1)
        if len(w_reshaped) > 10:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(w_reshaped)
            threshold = np.mean(kmeans.cluster_centers_)
            clean_weights = np.where(w < threshold, 0, w)
        else:
            clean_weights = w

        if np.sum(clean_weights) > 0:
            clean_weights /= np.sum(clean_weights)

        return clean_weights


class RobustDetector:
    def __init__(self, alpha_range=None):
        self.alpha_range = alpha_range or [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 0.01]
        self.best_alpha = None
        self.model = None

    def fit(self, X, y, sample_weights):
        X_inner_train, X_inner_val, y_inner_train, y_inner_val, w_inner_train, w_inner_val = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42
        )

        best_score = float('inf')
        best_alpha = self.alpha_range[1]

        if len(X_inner_train) > 50:
            for alpha in self.alpha_range:
                temp_model = SGDRegressor(
                    loss='huber', penalty='l1', alpha=alpha, epsilon=1.35,
                    max_iter=1000, tol=1e-3, learning_rate='adaptive', eta0=0.01, random_state=42
                )
                temp_model.fit(X_inner_train, y_inner_train, sample_weight=w_inner_train)
                preds = temp_model.predict(X_inner_val)
                residuals_sq = (y_inner_val - preds) ** 2
                weighted_mse = np.average(residuals_sq, weights=w_inner_val)

                if weighted_mse < best_score:
                    best_score = weighted_mse
                    best_alpha = alpha

            print(f"      >> Best Alpha: {best_alpha} (Weighted Val Loss: {best_score:.6f})")
        else:
            print("      >> [Warning] Sample too small for tuning, using default alpha.")

        self.best_alpha = best_alpha
        self.model = SGDRegressor(
            loss='huber', penalty='l1', alpha=self.best_alpha, epsilon=1.35,
            max_iter=3000, tol=1e-4, learning_rate='adaptive', eta0=0.01, random_state=42
        )

        if np.sum(sample_weights) > 0:
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)

    def predict_score(self, X, y):
        return np.abs(y - self.model.predict(X))


class BaselineComparator:
    def __init__(self):
        pass

    def run_isolation_forest(self, X_train, y_train, X_test, y_test):
        iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        preds = iso.fit_predict(X_train)
        mask = preds == 1

        clean_model = SGDRegressor(loss='huber', random_state=42)
        if np.sum(mask) > 10:
            clean_model.fit(X_train[mask], y_train[mask])
        else:
            clean_model.fit(X_train, y_train)

        return np.abs(y_test - clean_model.predict(X_test))

    def run_ransac(self, X_train, y_train, X_test, y_test):
        ransac = RANSACRegressor(
            estimator=SGDRegressor(loss='squared_error', random_state=42),
            min_samples=0.6,
            random_state=42
        )
        ransac.fit(X_train, y_train)
        return np.abs(y_test - ransac.predict(X_test))

# ==========================================
# 3. Utility & Plotting Functions
# ==========================================

def run_adaptive_attack_experiment(dataset_name, detector, X_test, y_test, y_true):
    attack_indices = np.where(y_true == 1)[0]
    if len(attack_indices) == 0:
        print("   [Skip] No attack samples to simulate.")
        return

    if len(attack_indices) > 5000:
        attack_indices = np.random.choice(attack_indices, 5000, replace=False)

    X_attack = X_test[attack_indices]
    y_attack_actual_log = y_test[attack_indices]
    y_attack_required_log = detector.model.predict(X_attack)

    duration_actual = np.maximum(np.expm1(y_attack_actual_log), 1e-6)
    duration_required = np.maximum(np.expm1(y_attack_required_log), 1e-6)
    rate_reduction_factor = duration_required / duration_actual
    valid_mask = (duration_required > duration_actual * 0.9) & (rate_reduction_factor < 1e9)

    if np.sum(valid_mask) > 10:
        median_factor = np.median(rate_reduction_factor[valid_mask])
        print(f"   [{dataset_name}] RATE PARADOX ANALYSIS:")
        print(f"      >> Median Duration Inflation Required: {median_factor:.2f}x (Slower)")
        print(f"      >> Attack Effectiveness Drop:          {(1 - 1 / median_factor) * 100:.2f}%")
    else:
        valid_mask[:100] = True

    plt.figure(figsize=(7, 6))
    plt.scatter(duration_actual[valid_mask], duration_required[valid_mask],
                alpha=0.4, c='red', label='Adaptive Attack Samples', s=30, edgecolors='darkred', linewidth=0.5)

    if np.sum(valid_mask) > 0:
        max_val = max(np.max(duration_actual[valid_mask]), np.max(duration_required[valid_mask]))
        min_val = min(np.min(duration_actual[valid_mask]), np.min(duration_required[valid_mask]))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Original Attack Baseline')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Original Attack Duration (Fast) [Log s]')
    plt.ylabel('Required Evasive Duration (Slow) [Log s]')
    plt.title(f'The Rate Paradox: Evasion Cost on {dataset_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    save_path = f'./results/Fig5_Rate_Paradox_{dataset_name}.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_linearization_effect(dataset_name, df, sample_n=2000):
    df_sample = df.sample(n=sample_n, random_state=42) if len(df) > sample_n else df.copy()

    col_dur = next((c for c in df.columns if 'Duration' in c or 'dur' in c), None)
    col_pkts = next((c for c in df.columns if 'Total' in c and 'Packets' in c or 'pkts' in c), None)

    if not col_dur or not col_pkts:
        return

    benign = df_sample[df_sample['Is_Attack'] == 0]
    attack = df_sample[df_sample['Is_Attack'] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(benign[col_pkts], benign[col_dur], c='green', alpha=0.5, s=15, label='Benign')
    axes[0].scatter(attack[col_pkts], attack[col_dur], c='red', alpha=0.6, s=15, marker='x', label='Attack')
    axes[0].set_xlabel("Packet Count (Raw)", fontsize=12)
    axes[0].set_ylabel("Flow Duration (Raw)", fontsize=12)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x_b_log, y_b_log = np.log1p(benign[col_pkts]), np.log1p(benign[col_dur])
    x_a_log, y_a_log = np.log1p(attack[col_pkts]), np.log1p(attack[col_dur])

    axes[1].scatter(x_b_log, y_b_log, c='green', alpha=0.5, s=15, label='Benign')
    axes[1].scatter(x_a_log, y_a_log, c='red', alpha=0.6, s=15, marker='x', label='Attack')

    if len(x_b_log) > 0:
        m, b = np.polyfit(x_b_log, y_b_log, 1)
        x_range = np.linspace(x_b_log.min(), x_b_log.max(), 100)
        axes[1].plot(x_range, m * x_range + b, 'k--', lw=1.5, label='Physical Invariant Line')

    axes[1].set_xlabel("Log(Packet Count)", fontsize=12)
    axes[1].set_ylabel("Log(Flow Duration)", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'Fig1_Linearization_{dataset_name}.pdf'), bbox_inches='tight')
    plt.close()


def plot_spectral_analysis(dataset_name, view_name, raw_weights, y_true):
    if len(raw_weights) < 50: return

    w_reshaped = raw_weights.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(w_reshaped)
    threshold = np.mean(kmeans.cluster_centers_)

    w_benign = raw_weights[y_true == 0]
    w_attack = raw_weights[y_true == 1]

    plt.figure(figsize=(8, 5))
    sns.kdeplot(w_benign, fill=True, color='green', label='Benign', warn_singular=False, cut=0)
    sns.kdeplot(w_attack, fill=True, color='red', label='Attack', warn_singular=False, cut=0)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Adaptive Cutoff ({threshold:.1e})')
    plt.yscale('log')
    plt.xlabel('Learned Spectral Weight')
    plt.ylabel('Density (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, f'Fig2_Spectral_Density_{dataset_name}_{view_name}.pdf'), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    n_samples = len(raw_weights)
    idx_sample = np.random.choice(n_samples, min(n_samples, 1000), replace=False)
    w_sample, y_sample_true = raw_weights[idx_sample], y_true[idx_sample]

    plt.scatter(w_sample[y_sample_true == 0], np.random.normal(1, 0.1, np.sum(y_sample_true == 0)),
                s=15, c='green', alpha=0.5, label='Benign')
    plt.scatter(w_sample[y_sample_true == 1], np.random.normal(1, 0.1, np.sum(y_sample_true == 1)),
                s=15, c='red', alpha=0.6, marker='x', label='Attack')

    plt.axvline(threshold, color='blue', linewidth=2, label='Cutoff Boundary')
    plt.text(threshold * 0.5, 1.3, "Eviction Region", color='red', ha='center', fontweight='bold')
    plt.text(threshold * 1.5, 1.3, "Retained Region", color='green', ha='center', fontweight='bold')
    plt.xlabel('Spectral Weight')
    plt.yticks([])
    plt.legend(loc='lower right')
    plt.grid(True, axis='x', alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, f'Fig3_Adaptive_Cutoff_{dataset_name}_{view_name}.pdf'), bbox_inches='tight')
    plt.close()


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    for dataset_name, config in DATASETS_CONFIG.items():
        print(f"\n{'#'*60}\n STARTING EXPERIMENT FOR DATASET: {dataset_name}\n{'#'*60}")

        input_file, poison_ratio = config['path'], config['poison_ratio']
        print(f"[1] Loading Data from {input_file}...")
        
        loader = ThreatModelLoader(input_file, poison_ratio=poison_ratio)
        df_all = loader.load_and_poison()
        
        if df_all.empty:
            continue

        plot_linearization_effect(dataset_name, df_all)

        df_train_full, df_test = train_test_split(df_all, test_size=0.2, stratify=df_all['Is_Attack'], random_state=42)
        df_train, df_val = train_test_split(df_train_full, test_size=0.25, stratify=df_train_full['Is_Attack'], random_state=42)

        print(f"   Train Set: {len(df_train)}\n   Val Set:   {len(df_val)}\n   Test Set:  {len(df_test)}")

        VIEWS = {'Time': 'log_Flow Duration', 'Volume': 'log_Total Length of Fwd Packets'}
        scores = {'val': {'y_true': df_val['Is_Attack'].values}, 'test': {'y_true': df_test['Is_Attack'].values}, 'baselines': {}}

        print("\n[2] Training Robust Models per View...")
        detectors, engineers = {}, {}

        for v_name, target in VIEWS.items():
            print(f"   > View: {v_name}")

            eng = FeatureEngineer(target).fit(df_train)
            engineers[v_name] = eng
            X_train, y_train = eng.transform(df_train)
            X_val, y_val = eng.transform(df_val)
            X_test, y_test = eng.transform(df_test)

            sanitizer = SpectralSanitizer()
            train_weights = sanitizer.get_weights(X_train)
            plot_spectral_analysis(dataset_name, v_name, train_weights, df_train['Is_Attack'].values)

            detector = RobustDetector()
            detector.fit(X_train, y_train, train_weights)
            detectors[v_name] = detector

            resid_scaler = StandardScaler().fit(detector.predict_score(X_train, y_train).reshape(-1, 1))
            scores['val'][v_name] = resid_scaler.transform(detector.predict_score(X_val, y_val).reshape(-1, 1)).ravel()
            scores['test'][v_name] = resid_scaler.transform(detector.predict_score(X_test, y_test).reshape(-1, 1)).ravel()

            print(f"      Running Baselines for View: {v_name}...")
            comparator = BaselineComparator()
            
            base_iso_resid = comparator.run_isolation_forest(X_train, y_train, X_test, y_test)
            scores_iso = StandardScaler().fit_transform(base_iso_resid.reshape(-1, 1)).ravel()
            
            base_ransac_resid = comparator.run_ransac(X_train, y_train, X_test, y_test)
            scores_ransac = StandardScaler().fit_transform(base_ransac_resid.reshape(-1, 1)).ravel()

            scores['baselines'][v_name] = {'iForest': scores_iso, 'RANSAC': scores_ransac}
            print(f"      Val AUC ({v_name}): {roc_auc_score(scores['val']['y_true'], scores['val'][v_name]):.4f}")

        print("\n[3] Optimizing Dual-View Fusion...")
        w_range = np.linspace(0, 1, 21)
        best_auc, best_w = 0, 0.5

        for w in w_range:
            fused = w * scores['val']['Time'] + (1 - w) * scores['val']['Volume']
            current_auc = roc_auc_score(scores['val']['y_true'], fused)
            if current_auc > best_auc:
                best_auc, best_w = current_auc, w

        print(f"   Best Weight: w_Time={best_w:.2f}, w_Vol={1 - best_w:.2f}")

        print("\n[4] Final Evaluation & Generating Figures...")
        y_test_true = scores['test']['y_true']
        final_scores = best_w * scores['test']['Time'] + (1 - best_w) * scores['test']['Volume']
        final_auc = roc_auc_score(y_test_true, final_scores)

        print("=" * 40)
        print(f"[{dataset_name}] ROBUSTNESS RESULT (Test Set): AUC = {final_auc:.4f}")

        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test_true, final_scores)
        plt.plot(fpr, tpr, color='red', lw=2, label=f'RoS-ETA (Ours, AUC={auc(fpr, tpr):.3f})')

        def get_baseline_fusion(method_key):
            available = [scores['baselines'][v][method_key] for v in ['Time', 'Volume']]
            return np.mean(available, axis=0)

        for name, color, style in [('iForest', 'green', '--'), ('RANSAC', 'blue', '-.')]:
            b_score = get_baseline_fusion(name)
            fpr_b, tpr_b, _ = roc_curve(y_test_true, b_score)
            plt.plot(fpr_b, tpr_b, color=color, linestyle=style, lw=2, label=f'{name} (AUC={auc(fpr_b, tpr_b):.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':', alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'ROC Comparison on {dataset_name}')
        plt.legend(loc="lower right", frameon=True, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, f'RoS_ETA_ROC_{dataset_name}_Comparison.pdf'), bbox_inches='tight')
        plt.close()

        print(f"   [Analysis] Generating Rate Paradox Graph for {dataset_name} using FULL DATASET...")
        X_full_time, y_full_time = engineers['Time'].transform(df_all)
        run_adaptive_attack_experiment(dataset_name, detectors['Time'], X_full_time, y_full_time, df_all['Is_Attack'].values)

        plt.figure(figsize=(8, 8))
        idx = np.random.choice(len(y_test_true), size=min(1000, len(y_test_true)), replace=False)
        plt.scatter(scores['test']['Time'][idx][y_test_true[idx] == 0], scores['test']['Volume'][idx][y_test_true[idx] == 0],
                    c='green', alpha=0.5, label='Benign', s=20)
        plt.scatter(scores['test']['Time'][idx][y_test_true[idx] == 1], scores['test']['Volume'][idx][y_test_true[idx] == 1],
                    c='red', alpha=0.6, label='Poison Attack', s=20, marker='x')
        plt.xlabel('Standardized Residual (Time View)')
        plt.ylabel('Standardized Residual (Volume View)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, f'RoS_ETA_Boundary_{dataset_name}.pdf'), bbox_inches='tight')
        plt.close()

        print(f"[{dataset_name}] Generating Ablation Study Plot (Fig 9)...")
        methods = ['Time View', 'Volume View', 'RoS-ETA (Fusion)']
        aucs = [roc_auc_score(y_test_true, scores['test']['Time']), roc_auc_score(y_test_true, scores['test']['Volume']), final_auc]
        
        plt.figure(figsize=(6, 5))
        bars = plt.bar(methods, aucs, color=['skyblue', 'lightgreen', '#ff6f61'], edgecolor='black', alpha=0.8)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            if height < 0.5: plt.text(i, height + 0.1, "FAILURE!", color='red', ha='center', fontweight='bold')

        plt.ylim([0.0, 1.1])
        plt.ylabel('AUC Score', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.savefig(os.path.join(RESULTS_DIR, f'Fig9_Ablation_{dataset_name}.pdf'), bbox_inches='tight')
        plt.close()

    print("\nAll experiments completed.")
