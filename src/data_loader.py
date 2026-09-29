import os
import glob
import re
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DATA_DIR = BASE_DIR / "data" / "original_samples"
CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned_samples"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed_graphs"

# Comprehensive Case/Dash-Insensitive MITRE ATT&CK Mapping
LABEL_MAPPING = {
    'benign': 0,
    'dos hulk': 3, 'dos goldeneye': 3, 'dos slowloris': 3, 'dos slowhttptest': 3, 'ddos': 3,
    'portscan': 1, 'ftp-patator': 1, 'ssh-patator': 1,
    'bot': 2, 'infiltration': 2, 'web attack - brute force': 2, 'web attack - xss': 2, 'web attack - sql injection': 2,
    'heartbleed': 4
}

NUMERIC_FEATURE_COLS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Inbound'
]

def sanitize_label_string(label_str):
    """Replaces non-ASCII dashes/dashes hex codes with standard hyphens."""
    if not isinstance(label_str, str):
        return 'benign'
    # Clean non-ASCII hyphens/dashes commonly present in CIC-IDS2017
    cleaned = re.sub(r'[\x96\x97\u2013\u2014]', '-', label_str).strip().lower()
    return cleaned

def load_csv_safely(file_path_or_buffer, nrows=None):
    try:
        df = pd.read_csv(file_path_or_buffer, nrows=nrows, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)
        df = pd.read_csv(file_path_or_buffer, nrows=nrows, encoding='latin1', low_memory=False)
    
    df.columns = df.columns.str.strip()
    return df

def clean_and_combine_raw_csvs(max_rows_per_file=50000):
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    master_path = CLEANED_DATA_DIR / "master_dataset.csv"
    
    csv_files = glob.glob(str(ORIGINAL_DATA_DIR / "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ORIGINAL_DATA_DIR}")
        
    df_list = []
    print(f"Ingesting {len(csv_files)} raw CSV file(s)...")

    for file_path in csv_files:
        try:
            chunk = load_csv_safely(file_path, nrows=max_rows_per_file)
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(inplace=True)
            if 'Label' in chunk.columns:
                df_list.append(chunk)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    if not df_list:
        raise ValueError("No valid flow records loaded.")

    master_df = pd.concat(df_list, ignore_index=True)
    master_df.to_csv(master_path, index=False)
    return master_df

def csv_to_graph_snapshots(file_path_or_buffer, window_size=300, target_features=79):
    df = load_csv_safely(file_path_or_buffer)
    
    feature_cols = [col for col in NUMERIC_FEATURE_COLS if col in df.columns]
    
    # 1. Fill NaNs and Infinities in raw data
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2. Safe Normalization (Prevents 0.0 division NaN generation)
    means = df[feature_cols].mean()
    stds = df[feature_cols].std().fillna(0.0)
    stds[stds == 0.0] = 1.0  # Safeguard against zero std columns
    
    df[feature_cols] = (df[feature_cols] - means) / stds
    df[feature_cols] = df[feature_cols].fillna(0.0)

    snapshots = []
    total_windows = len(df) // window_size

    if total_windows == 0 and len(df) > 0:
        total_windows = 1
        window_size = len(df)

    for i in range(total_windows):
        window = df.iloc[i * window_size : (i + 1) * window_size]
        if window.empty:
            continue
            
        if 'Destination Port' in window.columns:
            dst_ports = window['Destination Port'].values.astype(int)
            unique_nodes, node_indices = np.unique(dst_ports, return_inverse=True)
            num_nodes = len(unique_nodes)
            
            if num_nodes > 1:
                src_nodes = node_indices[:-1]
                dst_nodes = node_indices[1:]
            else:
                num_nodes = 5
                src_nodes = np.array([0, 1, 2, 3])
                dst_nodes = np.array([1, 2, 3, 4])
                node_indices = np.zeros(len(window), dtype=int)
        else:
            num_nodes = 10
            src_nodes = np.arange(num_nodes - 1)
            dst_nodes = np.arange(1, num_nodes)
            node_indices = np.zeros(len(window), dtype=int)

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        
        num_present_features = len(feature_cols)
        x = torch.zeros((num_nodes, target_features), dtype=torch.float)
        
        for idx in range(num_nodes):
            mask = (node_indices == idx)
            port_flows = window[feature_cols].values[mask]
            if len(port_flows) > 0:
                mean_feats = np.nan_to_num(port_flows.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
                fill_dim = min(num_present_features, target_features)
                x[idx, :fill_dim] = torch.tensor(mean_feats[:fill_dim], dtype=torch.float)

        edge_attr = torch.zeros((edge_index.shape[1], target_features), dtype=torch.float)

        if 'Label' in window.columns:
            raw_labels = window['Label'].values
            mapped_labels = [LABEL_MAPPING.get(sanitize_label_string(lbl), 0) for lbl in raw_labels]
            dominant_label = max(mapped_labels)
        else:
            dominant_label = 0
            
        y = torch.tensor([dominant_label], dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        snapshots.append(graph)

    return snapshots