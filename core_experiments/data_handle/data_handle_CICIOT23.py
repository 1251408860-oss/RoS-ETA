import pandas as pd
import numpy as np
import os
import time
import warnings

warnings.filterwarnings('ignore')

INPUT_FILE = '../CICIOT23_data/train/train.csv'
OUTPUT_FILE = '../CICIOT23_data/CICIOT23_Processed_Lite.csv'
CHUNK_SIZE = 500000

def preprocess_chunk(df_chunk):
    df_chunk.columns = df_chunk.columns.str.strip()

    if 'label' in df_chunk.columns:
        df_chunk['Label'] = df_chunk['label']
        df_chunk['Is_Attack'] = df_chunk['label'].apply(lambda x: 0 if 'Benign' in str(x) else 1)

    if 'flow_duration' in df_chunk.columns:
        df_chunk['Flow Duration'] = df_chunk['flow_duration'].replace(0, 1)

    if 'Tot size' in df_chunk.columns:
        df_chunk['Total Length of Fwd Packets'] = df_chunk['Tot size']

    if 'Tot sum' in df_chunk.columns:
        df_chunk['Total Fwd Packets'] = df_chunk['Tot sum']

    target_cols = ['Flow Duration', 'Total Length of Fwd Packets', 'Total Fwd Packets']

    for col in target_cols:
        if col in df_chunk.columns:
            clean_col = df_chunk[col].clip(lower=0)
            df_chunk[f'log_{col}'] = np.log1p(clean_col)

    keep_cols = ['Is_Attack', 'Label',
                 'log_Flow Duration', 'log_Total Length of Fwd Packets', 'log_Total Fwd Packets']
    
    final_cols = [c for c in keep_cols if c in df_chunk.columns]
    
    return df_chunk[final_cols]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print(f"Processing: {INPUT_FILE}")
    print(f"Output path: {OUTPUT_FILE}")

    start_time = time.time()
    total_rows = 0
    header_written = False

    with pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE) as reader:
        for chunk in reader:
            processed = preprocess_chunk(chunk)
            
            mode = 'a' if header_written else 'w'
            processed.to_csv(OUTPUT_FILE, mode=mode, index=False, header=not header_written)
            
            header_written = True
            total_rows += len(processed)
            print(f"Processed {total_rows} rows...", end='\r')

    print(f"\nProcessing complete! Total rows: {total_rows}")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
