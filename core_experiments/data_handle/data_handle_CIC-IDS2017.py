import pandas as pd
import numpy as np
import glob
import os
import time
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = '../CIC-IDS2017_data'
INPUT_PATTERN = os.path.join(DATA_DIR, '*.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'CIC-IDS2017_Processed_Lite.csv')
CHUNK_SIZE = 50000 

def preprocess_chunk(df_chunk):
    df_chunk.columns = df_chunk.columns.str.strip()
    
    df_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_chunk.dropna(inplace=True)
    
    if 'Label' in df_chunk.columns:
        df_chunk['Is_Attack'] = df_chunk['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
    
    if 'Flow Duration' in df_chunk.columns:
        df_chunk['Flow Duration'] = df_chunk['Flow Duration'].replace(0, 1)
    
    if 'Total Length of Fwd Packets' in df_chunk.columns:
        df_chunk['Bytes_per_sec'] = df_chunk['Total Length of Fwd Packets'] / df_chunk['Flow Duration']
    
    if 'Total Fwd Packets' in df_chunk.columns:
        df_chunk['Packets_per_sec'] = df_chunk['Total Fwd Packets'] / df_chunk['Flow Duration']
    
    numeric_cols = df_chunk.select_dtypes(include=[np.number]).columns
    cols_to_exclude = ['Is_Attack', 'Label', 'Destination Port', 'Source Port', 'Protocol']
    cols_to_log = [c for c in numeric_cols if c not in cols_to_exclude]
    
    for col in cols_to_log:
        if (df_chunk[col] >= 0).all():
            df_chunk[f'log_{col}'] = np.log1p(df_chunk[col])
            
    return df_chunk

def main():
    all_files = glob.glob(INPUT_PATTERN)
    all_files = [f for f in all_files if os.path.basename(f) != os.path.basename(OUTPUT_FILE)]
    
    print(f"Detected {len(all_files)} source files. Output path: {OUTPUT_FILE}")
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    start_time = time.time()
    total_rows = 0
    header_written = False

    for filename in all_files:
        print(f"--> Processing file: {os.path.basename(filename)}")
        
        chunk_iter = pd.read_csv(filename, chunksize=CHUNK_SIZE, encoding='cp1252', low_memory=False)
        
        for chunk in chunk_iter:
            processed_chunk = preprocess_chunk(chunk)
            
            if not header_written:
                processed_chunk.to_csv(OUTPUT_FILE, mode='w', index=False)
                header_written = True
            else:
                processed_chunk.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)
            
            total_rows += len(processed_chunk)

    end_time = time.time()
    print(f"\nProcessing complete! Cleaned data saved to: {OUTPUT_FILE}")
    print(f"Total time: {end_time - start_time:.2f} seconds | Total rows: {total_rows}")

if __name__ == "__main__":
    main()
