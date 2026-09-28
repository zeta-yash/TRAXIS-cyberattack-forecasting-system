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

def csv_to_graph_snapshots(df, window_size=300):
    """
    Converts a dataframe into a sequence of PyG Data objects.
    """
    df = clean_dataframe(df)
    
    # Extract numerical telemetry columns
    ignore_cols = ['Label', 'Source IP', 'Destination IP', 'Timestamp', 'Flow ID']
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    # Label Column identification
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    
    num_windows = len(df) // window_size
    snapshots = []

    for i in range(num_windows):
        sub_df = df.iloc[i * window_size : (i + 1) * window_size]
        
        # Determine host/port nodes
        if 'Destination Port' in sub_df.columns:
            dst_ports = sub_df['Destination Port'].astype(int).values
            unique_nodes, node_mapping = np.unique(dst_ports, return_inverse=True)
            num_nodes = len(unique_nodes)
            
            # Construct directed edge index (sequential flow connections)
            src_indices = np.arange(len(sub_df) - 1) % num_nodes
            dst_indices = node_mapping[1:]
            edge_index = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)
        else:
            num_nodes = 20
            edge_index = torch.randint(0, num_nodes, (2, window_size), dtype=torch.long)

        # Feature matrix (N, F)
        raw_feats = sub_df[feature_cols].values
        # Normalize features
        mean = np.mean(raw_feats, axis=0)
        std = np.std(raw_feats, axis=0) + 1e-6
        norm_feats = (raw_feats - mean) / std

        # Node feature aggregation
        node_features = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        for idx in range(min(len(sub_df), num_nodes)):
            node_features[idx % num_nodes] = norm_feats[idx]

        # Edge feature matrix
        edge_attr = torch.tensor(norm_feats[:edge_index.shape[1]], dtype=torch.float)

        # Determine target MITRE stage for the window
        raw_labels = sub_df[label_col].values
        mitre_stages = [map_label_to_mitre(lbl) for lbl in raw_labels]
        # Assign highest severity stage observed in window
        window_label = max(mitre_stages, key=lambda x: (x if x != 0 else -1))

        x_tensor = torch.tensor(node_features, dtype=torch.float)
        y_tensor = torch.tensor([window_label], dtype=torch.long)

        graph_data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        snapshots.append(graph_data)

    return snapshots