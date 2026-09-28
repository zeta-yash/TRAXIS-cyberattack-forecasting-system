import sys
from pathlib import Path

# Add project root to Python search path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import os
import torch
import pandas as pd
from src.data_loader import make_master_dataset, csv_to_graph_snapshots

def build_processed_dataset():
    RAW_DIR = BASE_DIR / "data" / "original_samples"
    CLEANED_CSV_PATH = BASE_DIR / "data" / "cleaned_samples" / "master_dataset.csv"
    PROCESSED_GRAPH_PATH = BASE_DIR / "data" / "processed_graphs" / "master_graphs.pt"

    os.makedirs(CLEANED_CSV_PATH.parent, exist_ok=True)
    os.makedirs(PROCESSED_GRAPH_PATH.parent, exist_ok=True)

    # 1. Clean & Consolidate Master CSV
    if RAW_DIR.exists() and list(RAW_DIR.glob("*.csv")):
        master_df = make_master_dataset(RAW_DIR, CLEANED_CSV_PATH)
    elif CLEANED_CSV_PATH.exists():
        print(f"[PrepareDataset] Loading existing master CSV from {CLEANED_CSV_PATH}")
        master_df = pd.read_csv(CLEANED_CSV_PATH)
    else:
        raise FileNotFoundError(f"Neither raw CSVs in {RAW_DIR} nor {CLEANED_CSV_PATH} exist!")

    # 2. Build Graph Snapshots
    print("[PrepareDataset] Converting telemetry flows into PyG graph snapshots...")
    snapshots = csv_to_graph_snapshots(master_df, window_size=300)

    # 3. Serialize Graph Tensor
    torch.save(snapshots, PROCESSED_GRAPH_PATH)
    print(f"[PrepareDataset] Successfully generated {len(snapshots)} snapshots -> {PROCESSED_GRAPH_PATH}")
    return PROCESSED_GRAPH_PATH

if __name__ == "__main__":
    build_processed_dataset()