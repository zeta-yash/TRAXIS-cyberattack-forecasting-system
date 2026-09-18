import os
import sys

import pandas as pd
import numpy as np

RAW_INPUT_PATH = os.path.join("data", "02-14-2018.csv") #for testing purpose onlyy..

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df

# =====================================================================
# 1. Pipeline Dev Slice (20,000 - 50,000 rows, Stratified)
# Purpose: Instant loading (<2 sec), testing PyG tensors & node mappings
# =====================================================================
def make_dev_slice(input_path: str = RAW_INPUT_PATH, output_path: str = "data/dev_sample_30k.csv", n_samples: int = 30000):
    print("Generating Pipeline Dev Slice...")
    # Read in chunks to keep RAM footprint negligible on 16GB Mac
    chunk_size = 50000
    chunks = []
    
    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        chunk = clean_columns(chunk)
        chunks.append(chunk)
        if sum(len(c) for c in chunks) >= 150000:
            break
            
    df_pool = pd.concat(chunks, ignore_index=True)
    
    # Stratified sample across all available attack and benign classes
    dev_slice = (
        df_pool.groupby("Label", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), n_samples // df_pool["Label"].nunique()), random_state=42))
        .reset_index(drop=True)
    )
    
    dev_slice.to_csv(output_path, index=False)
    print(f"Saved Dev Slice ({len(dev_slice)} rows) -> {output_path}")

# =====================================================================
# 2. Single-Attack Benchmark Slice (100,000 - 300,000 rows, Temporal)
# Purpose: Realistic continuous traffic sequence for GNN + Temporal baseline
# =====================================================================
def make_benchmark_slice(input_path: str, output_path: str = "data/benchmark_ddos_150k.csv", n_rows: int = 150000):
    print("Generating Single-Attack Benchmark Slice...")
    
    # Read the top n continuous rows directly
    df = pd.read_csv(input_path, nrows=n_rows, low_memory=False)
    df = clean_columns(df)
    
    # If Timestamp column exists, ensure chronological order for temporal modeling
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp")
        
    df.to_csv(output_path, index=False)
    print(f"Saved Benchmark Slice ({len(df)} rows) -> {output_path}")

# =====================================================================
# 3. Full Cleaned Dataset (Downcasted Memory Footprint)
# Purpose: Production/final runs without crashing 16GB unified memory
# =====================================================================
def make_optimized_full_dataset(input_path: str, output_path: str = "data/full_optimized.csv"):
    print("Optimizing Full Dataset (Downcasting float64/int64)...")
    
    first_chunk = True
    for chunk in pd.read_csv(input_path, chunksize=100000, low_memory=False):
        chunk = clean_columns(chunk)
        
        # Replace infinities and drop NaNs
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Downcast 64-bit numbers to 32-bit to halve memory usage
        for col in chunk.select_dtypes(include=["float64"]).columns:
            chunk[col] = chunk[col].astype(np.float32)
        for col in chunk.select_dtypes(include=["int64"]).columns:
            chunk[col] = chunk[col].astype(np.int32)
            
        # Append chunk to output file
        chunk.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        
    print(f"Saved Optimized Full Dataset -> {output_path}")

if __name__ == "__main__":
    make_dev_slice(RAW_INPUT_PATH)
    #to create testing level file
    # make_benchmark_slice(RAW_INPUT_PATH) 
    # Uncomment only when you are ready to prepare the entire dataset
    # make_optimized_full_dataset(RAW_INPUT_PATH)