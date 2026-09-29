# SHAP attribution

import torch
import numpy as np
import pandas as pd

def extract_feature_importance(graph_snapshot, feature_names=None):
    """
    Computes simple feature importance for a graph snapshot 
    by calculating average magnitude across node and edge attributes.
    """
    if graph_snapshot.edge_attr is not None:
        # Mean intensity across all flow edges in the current time window
        importance_scores = torch.mean(torch.abs(graph_snapshot.edge_attr), dim=0).cpu().numpy()
    else:
        importance_scores = torch.mean(torch.abs(graph_snapshot.x), dim=0).cpu().numpy()

    if feature_names is None or len(feature_names) != len(importance_scores):
        feature_names = [f"Telemetry_Feature_{i}" for i in range(len(importance_scores))]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=False)

    return importance_df

if __name__ == "__main__":
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    sample_path = BASE_DIR / "data" / "processed_graphs" / "dev_sample_30k_graphs.pt"
    
    if sample_path.exists():
        snapshots = torch.load(sample_path, weights_only=False)
        top_features = extract_feature_importance(snapshots[0])
        print("Top 5 Driving Features for Window 0:")
        print(top_features.head(5))