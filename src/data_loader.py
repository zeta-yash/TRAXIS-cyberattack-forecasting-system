import os
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data

def clean_dataframe(df):
    """
    Cleans raw flow dataframe headers and inf/nan values.
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Replace Infinity/NaN values with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

def map_label_to_mitre(label_str):
    """
    Maps raw attack strings to MITRE ATT&CK stages:
    0: Benign
    1: Reconnaissance / Initial Access
    2: Execution / Persistence
    3: Lateral Movement / DoS
    4: Exfiltration / Impact
    """
    lbl = str(label_str).upper()
    if 'BENIGN' in lbl:
        return 0
    elif any(k in lbl for k in ['PORTScan', 'RECON', 'PING']):
        return 1
    elif any(k in lbl for k in ['INFILTRATION', 'WEB ATTACK', 'XSS', 'SQL']):
        return 2
    elif any(k in lbl for k in ['DDOS', 'DOS', 'BOT']):
        return 3
    elif any(k in lbl for k in ['HEARTBLEED', 'PATATOR', 'SSH', 'FTP']):
        return 4
    else:
        return 0

def make_master_dataset(raw_dir, output_csv_path, samples_per_file=20000):
    """
    In-memory chunked reader for all raw CSV files in raw_dir.
    Creates a stratified master dataset at output_csv_path.
    """
    raw_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")

    processed_dfs = []
    print(f"[DataLoader] Ingesting {len(raw_files)} raw dataset files...")

    for fpath in raw_files:
        fname = os.path.basename(fpath)
        print(f"  └─ Processing: {fname}")
        
        # Read in chunks to prevent memory overload with encoding fallback
        chunks = []
        try:
            reader = pd.read_csv(fpath, chunksize=50000, low_memory=False, encoding_errors='replace')
            for chunk in reader:
                chunk = clean_dataframe(chunk)
                chunks.append(chunk)
                if sum(len(c) for c in chunks) >= samples_per_file * 2:
                    break
        except Exception as e:
            print(f"  ⚠️ Warning reading {fname}: {e}")
            continue
                
        if chunks:
            full_df = pd.concat(chunks, ignore_index=True)
            n_samples = min(len(full_df), samples_per_file)
            sample_df = full_df.sample(n=n_samples, random_state=42)
            processed_dfs.append(sample_df)

    if not processed_dfs:
        raise RuntimeError("No CSV files could be parsed successfully!")

    master_df = pd.concat(processed_dfs, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    master_df.to_csv(output_csv_path, index=False)
    print(f"[DataLoader] Saved consolidated master dataset ({len(master_df)} rows) -> {output_csv_path}")
    return master_df

def csv_to_graph_snapshots(df, window_size=100):
    """
    Converts a dataframe into a sequence of PyG Data objects.
    Dynamically adjusts window_size if dataframe size is small.
    """
    df = clean_dataframe(df)
    
    # Dynamically adjust window_size if dataframe is small
    if len(df) < window_size * 6:
        window_size = max(10, len(df) // 10)
    
    ignore_cols = ['Label', 'Source IP', 'Destination IP', 'Timestamp', 'Flow ID']
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    
    num_windows = len(df) // window_size
    snapshots = []

    for i in range(num_windows):
        sub_df = df.iloc[i * window_size : (i + 1) * window_size]
        
        if 'Destination Port' in sub_df.columns:
            dst_ports = sub_df['Destination Port'].astype(int).values
            unique_nodes, node_mapping = np.unique(dst_ports, return_inverse=True)
            num_nodes = max(len(unique_nodes), 2)
            
            src_indices = np.arange(len(sub_df) - 1) % num_nodes
            dst_indices = node_mapping[1:]
            edge_index = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)
        else:
            num_nodes = 20
            edge_index = torch.randint(0, num_nodes, (2, min(window_size, len(sub_df))), dtype=torch.long)

        raw_feats = sub_df[feature_cols].values
        mean = np.mean(raw_feats, axis=0)
        std = np.std(raw_feats, axis=0) + 1e-6
        norm_feats = (raw_feats - mean) / std

        node_features = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        for idx in range(min(len(sub_df), num_nodes)):
            node_features[idx % num_nodes] = norm_feats[idx]

        edge_attr = torch.tensor(norm_feats[:edge_index.shape[1]], dtype=torch.float)

        # Require majority/significant presence of attack flows (>10%)
        raw_labels = sub_df[label_col].values
        mitre_stages = [map_label_to_mitre(lbl) for lbl in raw_labels]
        non_benign = [s for s in mitre_stages if s != 0]
        
        if len(non_benign) > (len(sub_df) * 0.10):
            window_label = max(set(non_benign), key=non_benign.count)
        else:
            window_label = 0  # Pure Benign Baseline

        x_tensor = torch.tensor(node_features, dtype=torch.float)
        y_tensor = torch.tensor([window_label], dtype=torch.long)

        graph_data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        snapshots.append(graph_data)

    return snapshots