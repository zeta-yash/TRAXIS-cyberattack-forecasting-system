from pathlib import Path
import torch
from src.data_loader import clean_and_combine_raw_csvs, csv_to_graph_snapshots

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_MASTER_PATH = BASE_DIR / "data" / "cleaned_samples" / "master_dataset.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed_graphs" / "master_graphs.pt"

def build_master_dataset():
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not CLEANED_MASTER_PATH.exists():
        clean_and_combine_raw_csvs()
        
    print("Converting master dataset into dynamic graph snapshots...")
    snapshots = csv_to_graph_snapshots(CLEANED_MASTER_PATH, window_size=300)
    
    torch.save(snapshots, PROCESSED_DATA_PATH)
    print(f"Master dataset successfully processed! Saved {len(snapshots)} graph snapshots to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    build_master_dataset()