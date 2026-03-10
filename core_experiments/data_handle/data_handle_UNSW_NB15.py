import pandas as pd
import numpy as np
import glob
import os
import time
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = '../UNSW-NB15_data'
INPUT_PATTERN = os.path.join(DATA_DIR, 'UNSW-NB15_*.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'UNSW-NB15_Processed_Lite.csv')
CHUNK_SIZE = 50000

UNSW_COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt',
    'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
]

def preprocess_chunk(df_chunk):
    if 'Label' in df_chunk.columns:
        df_chunk['Is_Attack'] = df_chunk['Label'].astype(int)

    if 'dur' in df_chunk.columns:
        df_chunk['Flow Duration'] = df_chunk['dur'].replace(0, 1e-6) * 1_000_000

    if 'sbytes' in df_chunk.columns and 'dbytes' in df_chunk.columns:
        df_chunk['Total Length of Fwd Packets'] = df_chunk['sbytes'] + df_chunk['dbytes']

    if 'Spkts' in df_chunk.columns and 'Dpkts' in df_chunk.columns:
        df_chunk['Total Fwd Packets'] = df_chunk['Spkts'] + df_chunk['Dpkts']

    if 'Flow Duration' in df_chunk.columns and 'Total Length of Fwd Packets' in df_chunk.columns:
        duration_safe = df_chunk['Flow Duration'].replace(0, 1)
        df_chunk['Bytes_per_sec'] = df_chunk['Total Length of Fwd Packets'] / (duration_safe / 1e6) 

    target_cols = ['Flow Duration', 'Total Length of Fwd Packets', 'Total Fwd Packets']

    for col in target_cols:
        if col in df_chunk.columns:
            clean_col = df_chunk[col].clip(lower=0)
            df_chunk[f'log_{col}'] = np.log1p(clean_col)

    keep_cols = ['Is_Attack', 'Label', 'attack_cat',
                 'log_Flow Duration', 'log_Total Length of Fwd Packets', 'log_Total Fwd Packets']

    final_cols = [c for c in keep_cols if c in df_chunk.columns]

    return df_chunk[final_cols]


def main():
    all_files = glob.glob(INPUT_PATTERN)
    all_files = [f for f in all_files if 'Processed' not in f and 'features' not in f and 'LIST_EVENTS' not in f]

    target_files = [f for f in all_files if
                    f.endswith('1.csv') or f.endswith('2.csv') or f.endswith('3.csv') or f.endswith('4.csv')]

    has_header = None
    if not target_files:
        print("Warning: UNSW-NB15 1-4.csv not found, attempting to read training/testing sets...")
        target_files = [f for f in all_files if 'training-set' in f or 'testing-set' in f]
        has_header = 0

    print(f"Detected {len(target_files)} source files. Output path: {OUTPUT_FILE}")

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    start_time = time.time()
    total_rows = 0
    header_written = False

    for filename in target_files:
        print(f"--> Processing file: {os.path.basename(filename)}")

        if has_header is None:
            chunk_iter = pd.read_csv(filename, chunksize=CHUNK_SIZE, names=UNSW_COLUMNS, encoding='latin-1',
                                     low_memory=False)
        else:
            chunk_iter = pd.read_csv(filename, chunksize=CHUNK_SIZE, header=0, encoding='latin-1', low_memory=False)

        for chunk in chunk_iter:
            processed_chunk = preprocess_chunk(chunk)

            if not header_written:
                processed_chunk.to_csv(OUTPUT_FILE, mode='w', index=False)
                header_written = True
            else:
                processed_chunk.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

            total_rows += len(processed_chunk)
            print(f"   Processed {total_rows} rows...", end='\r')

    print(f"\nProcessing complete! Data saved to: {OUTPUT_FILE}")
    print(f"RoS-ETA Compatibility Check: Contains 'log_Flow Duration' and 'log_Total Length of Fwd Packets'")

if __name__ == "__main__":
    main()
