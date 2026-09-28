import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import NetworkAttackForecaster

def train_model():
    GRAPH_PATH = BASE_DIR / "data" / "processed_graphs" / "master_graphs.pt"
    MODEL_PATH = BASE_DIR / "models" / "pretrained_weights.pt"
    os.makedirs(MODEL_PATH.parent, exist_ok=True)

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Processed graph dataset not found at {GRAPH_PATH}. Run Step 1 first!")

    print(f"[Train] Loading graph snapshots from {GRAPH_PATH}...")
    snapshots = torch.load(GRAPH_PATH, weights_only=False)
    print(f"[Train] Loaded {len(snapshots)} snapshots.")

    k = 5  # Sequence length history (t-4, t-3, t-2, t-1, t) -> predict t+1
    if len(snapshots) <= k:
        raise ValueError(f"Need at least {k+1} snapshots to form training sequences.")

    sequences = []
    labels = []
    for i in range(len(snapshots) - k):
        seq = snapshots[i : i + k]
        target = snapshots[i + k].y.item()
        sequences.append(seq)
        labels.append(target)

    # # Calculate Inverse Class Weights for Imbalance
    # labels_tensor = torch.tensor(labels, dtype=torch.long)
    # class_counts = torch.bincount(labels_tensor, minlength=5).float()
    # print(f"[Train] MITRE Stage Distribution across windows: {class_counts.numpy().astype(int)}")

    # Calculate Dampened Class Weights
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor, minlength=5).float()
    print(f"[Train] MITRE Stage Distribution: {class_counts.numpy().astype(int)}")

    # Dampen inverse weights with square root so normal traffic maintains strong baseline presence
    class_weights = 1.0 / (torch.sqrt(class_counts) + 1e-5)
    class_weights = class_weights / class_weights.sum()
    print(f"[Train] Dampened Loss Weights: {class_weights.numpy().round(4)}")
    
    # Add small epsilon to prevent division by zero
    total_samples = len(labels)
    class_weights = total_samples / (5.0 * (class_counts + 1e-5))
    class_weights = class_weights / class_weights.sum()  # Normalize
    print(f"[Train] Computed Cross-Entropy Loss Weights: {class_weights.numpy().round(4)}")

    # Device Setup (MPS / CUDA / CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[Train] Using compute device: {device}")

    in_channels = snapshots[0].x.shape[1]
    model = NetworkAttackForecaster(in_channels=in_channels, hidden_channels=64, num_classes=5).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training Loop
    epochs = 20
    model.train()
    print("[Train] Starting GNN-GRU model training...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0

        for seq, target in zip(sequences, labels):
            # Move sequence graphs to target device
            seq_device = [data.to(device) for data in seq]
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)

            optimizer.zero_grad()
            out = model(seq_device)
            loss = criterion(out, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=-1).item()
            if pred == target:
                correct += 1

        acc = (correct / len(sequences)) * 100
        avg_loss = total_loss / len(sequences)
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    # Persist compact weights
    torch.save(model.state_dict(), MODEL_PATH)
    file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"[Train] Saved trained weights ({file_size_mb:.2f} MB) -> {MODEL_PATH}")

if __name__ == "__main__":
    train_model()