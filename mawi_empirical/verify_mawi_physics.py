import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CSV_FILE = 'mawi/mawi_full_day.csv'
CHUNK_SIZE = 1000000

def process_chunk(chunk):
    chunk.columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp', 'len']
    chunk['flow_id'] = chunk['src_ip'].astype(str) + chunk['dst_ip'].astype(str) + \
                       chunk['src_port'].astype(str) + chunk['dst_port'].astype(str)

    stats = chunk.groupby('flow_id').agg(
        duration=('timestamp', lambda x: x.max() - x.min()),
        total_bytes=('len', 'sum'),
        total_pkts=('len', 'count'),
        mean_len=('len', 'mean')
    )
    return stats

def run_big_data():
    print("=== RoS-ETA: MAWI Full-Day Big Data Validation ===")
    print(f"Reading and processing data in chunks: {CSV_FILE} ...")

    global_stats = []
    reader = pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE, iterator=True)

    for i, chunk in enumerate(reader):
        print(f"  Processing chunk {i + 1}...")
        stats = process_chunk(chunk)

        mask = (stats['duration'] > 0.1) & \
               (stats['total_bytes'] > 10000) & \
               (stats['total_pkts'] > 10)
        
        global_stats.append(stats[mask].copy())

    print("Merging chunk results...")
    if not global_stats:
        print("No valid data extracted. Exiting program.")
        return

    final_df = pd.concat(global_stats)
    final_df = final_df.groupby(final_df.index).agg({
        'duration': 'sum',
        'total_bytes': 'sum',
        'total_pkts': 'sum',
        'mean_len': 'mean'
    })

    print(f"Total number of valid physical flows extracted: {len(final_df)}")
    print("Performing RANSAC fitting and generating plot...")

    X = np.log10(final_df['total_bytes'].values)
    Y = np.log10(final_df['duration'].values)
    entropy_proxy = np.log1p(final_df['mean_len'].values)

    plt.figure(figsize=(12, 7))

    sc = plt.scatter(X, Y, c=entropy_proxy, cmap='viridis', alpha=0.5, s=10,
                     edgecolors='none', label='MAWI Full-Day Traffic', rasterized=True)
    cbar = plt.colorbar(sc)
    cbar.set_label('Micro-Entropy Proxy (Log Mean Len)', rotation=270, labelpad=15)

    reg = RANSACRegressor(random_state=42)
    reg.fit(X.reshape(-1, 1), Y)
    slope = reg.estimator_.coef_[0]
    intercept = reg.estimator_.intercept_

    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = slope * x_range + intercept
    plt.plot(x_range, y_pred, 'k--', linewidth=2, label=f'Benign Center (Slope={slope:.2f})')

    b_limit = np.percentile(Y - 1.0 * X, 0.1) - 0.5
    y_limit = 1.0 * x_range + b_limit
    plt.plot(x_range, y_limit, 'r-', linewidth=3, label='Physical Limit (Slope=1.0)')

    plt.fill_between(x_range, y_limit, y_pred, color='red', alpha=0.05)

    plt.xlabel(r'Log$_{10}$ (Traffic Volume $N$)', fontsize=14)
    plt.ylabel(r'Log$_{10}$ (Duration $T$)', fontsize=14)
    plt.title(f'Full-Day Physical Validation on MAWI Backbone (N={len(final_df)})', fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig('MAWI_Full_Day_Validation.pdf', dpi=300, bbox_inches='tight')
    print("Plot successfully generated: MAWI_Full_Day_Validation.pdf")

if __name__ == '__main__':
    run_big_data()
