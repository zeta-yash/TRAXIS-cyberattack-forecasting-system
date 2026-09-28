import os
import torch
from pathlib import Path
from data_loader import csv_to_graph_snapshots, make_dev_slice

# Configure directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned_samples"
PROCESSED_DIR = BASE_DIR / "data" / "processed_graphs"

def ensure_dev_slice_exists():
    """Generates the dev_sample_30k.csv file if cleaned_samples is empty."""
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    target_sample = CLEANED_DIR / "dev_sample_30k.csv"
    
    if not target_sample.exists():
        print(f"No cleaned samples found in {CLEANED_DIR}. Triggering dev slice creation...")
        make_dev_slice(output_path=str(target_sample))

def process_all_cleaned_samples(time_window_size=300):
    """Parses all CSVs in data/cleaned_samples/ into PyG graph tensors."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check/generate initial dev slice
    ensure_dev_slice_exists()

    cleaned_files = list(CLEANED_DIR.glob("*.csv"))

    if not cleaned_files:
        print(f"Error: No CSV files found in {CLEANED_DIR} even after slice check.")
        return

    print(f"\nFound {len(cleaned_files)} cleaned CSV file(s). Converting to graph snapshots...\n")

    for csv_path in cleaned_files:
        print(f"Processing: {csv_path.name}")
        snapshots = csv_to_graph_snapshots(str(csv_path), time_window_size=time_window_size)

        output_filename = f"{csv_path.stem}_graphs.pt"
        output_path = PROCESSED_DIR / output_filename

        torch.save(snapshots, output_path)
        print(f"-> Saved {len(snapshots)} graph snapshots to {output_path}\n")

if __name__ == "__main__":
    process_all_cleaned_samples()