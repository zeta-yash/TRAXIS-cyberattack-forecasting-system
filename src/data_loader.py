# uses telemetry - but for now we are using sample file

# and create csv

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

RAW_INPUT_PATH = os.path.join("data", "original_samples", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df

def map_label_to_attack_stage(label):
    """
    Map raw dataset labels to MITRE ATT&CK Stages (0 to 4).
    """
    label_str = str(label).upper()
    if 'BENIGN' in label_str:
        return 0  # Normal Traffic
    elif 'PORTSCAN' in label_str or 'RECON' in label_str:
        return 1  # Reconnaissance / Initial Access
    elif 'INFILTRATION' in label_str or 'WEB' in label_str or 'BOT' in label_str:
        return 2  # Execution / Persistence
    elif 'DOS' in label_str or 'DDOS' in label_str or 'BRUTE FORCE' in label_str:
        return 3  # Lateral Movement / Denial of Service
    else:
        return 4  # Exfiltration / Impact

# =====================================================================
# 1. Pipeline Dev Slice (20,000 - 50,000 rows, Stratified)
# Purpose: Instant loading (<2 sec), testing PyG tensors & node mappings
# =====================================================================
def make_dev_slice(input_path: str = RAW_INPUT_PATH, output_path: str = "data/cleaned_samples/dev_sample_30k.csv", n_samples: int = 30000):
    print("Generating Pipeline Dev Slice...")
    # Read in chunks to keep RAM footprint negligible 
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
        df_pool.groupby("Label")
        .apply(lambda x: x.sample(n=min(len(x), n_samples // df_pool["Label"].nunique()), random_state=42))
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    
    dev_slice.to_csv(output_path, index=False)
    print(f"Saved Dev Slice ({len(dev_slice)} rows) -> {output_path}")

# =====================================================================
# 2. Single-Attack Benchmark Slice (100,000 - 300,000 rows, Temporal)
# Purpose: Realistic continuous traffic sequence for GNN + Temporal baseline
# =====================================================================
def make_benchmark_slice(input_path: str, output_path: str = "data/cleaned_samples/benchmark_ddos_150k.csv", n_rows: int = 150000):
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
# =====================================================================
def make_optimized_full_dataset(input_path: str, output_path: str = "data/cleaned_samples/full_optimized.csv"):
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


# we have to create a parquet reader -> CSV converter for initial phase
# later we will be using telemetry packets

# ======================================================================
# 
#  ======================================================================
def csv_to_graph_snapshots(csv_path, time_window_size=300):
    """
    Converts a cleaned flow CSV into a sequence of PyTorch Geometric Data graph snapshots.
    """
    df = pd.read_csv(csv_path)
    df = clean_columns(df)
    
    # Handle infinite or missing values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Identify numerical feature columns
    ignore_cols = ['Source IP', 'Destination IP', 'Timestamp', 'Label', 'Flow ID']
    feature_cols = [c for c in df.columns if c not in ignore_cols and np.issubdtype(df[c].dtype, np.number)]
    
    # Normalize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Determine Source & Destination Nodes
    has_ips = 'Source IP' in df.columns and 'Destination IP' in df.columns
    
    snapshots = []
    num_windows = len(df) // time_window_size
    
    for i in range(num_windows):
        window_df = df.iloc[i * time_window_size : (i + 1) * time_window_size]
        
        if has_ips:
            unique_nodes = list(set(window_df['Source IP']).union(set(window_df['Destination IP'])))
            node_map = {ip: idx for idx, ip in enumerate(unique_nodes)}
            src_indices = [node_map[ip] for ip in window_df['Source IP']]
            dst_indices = [node_map[ip] for ip in window_df['Destination IP']]
        else:
            # Fallback if raw IP columns aren't present in the slice
            dst_port_col = 'Dst Port' if 'Dst Port' in window_df.columns else 'Destination Port'
            unique_nodes = list(window_df[dst_port_col].unique())
            node_map = {port: idx for idx, port in enumerate(unique_nodes)}
            dst_indices = [node_map[port] for port in window_df[dst_port_col]]
            src_indices = [(idx % len(unique_nodes)) for idx in range(len(window_df))]

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(window_df[feature_cols].values, dtype=torch.float)
        
        num_nodes = len(node_map)
        node_features = torch.zeros((num_nodes, len(feature_cols)), dtype=torch.float)
        for idx, src_node in enumerate(src_indices):
            node_features[src_node] += edge_attr[idx]
        
        window_labels = window_df['Label'].apply(map_label_to_attack_stage).values
        graph_label = torch.tensor([max(window_labels)], dtype=torch.long)
        
        graph_snapshot = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=graph_label,
            num_nodes=num_nodes
        )
        snapshots.append(graph_snapshot)

    return snapshots


if __name__ == "__main__":
    make_dev_slice(RAW_INPUT_PATH)
    #to create testing level file
    # make_benchmark_slice(RAW_INPUT_PATH) 
    # Uncomment only when you are ready to prepare the entire dataset
    # make_optimized_full_dataset(RAW_INPUT_PATH)

