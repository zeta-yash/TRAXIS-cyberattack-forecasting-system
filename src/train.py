import torch
import torch.nn as nn
from pathlib import Path
from src.model import NetworkAttackForecaster

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed_graphs" / "master_graphs.pt"
MODEL_SAVE_PATH = BASE_DIR / "models" / "pretrained_weights.pt"

def train():
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not PROCESSED_DATA_PATH.exists():
        from src.prepare_dataset import build_master_dataset
        build_master_dataset()

    snapshots = torch.load(PROCESSED_DATA_PATH, weights_only=False)
    print(f"Loaded {len(snapshots)} master graph snapshots for training.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")

    in_channels = snapshots[0].x.shape[1]
    model = NetworkAttackForecaster(in_channels=in_channels, hidden_channels=64, num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    seq_len = 3  # Stable context sequence length
    num_samples = len(snapshots) - seq_len

    if num_samples <= 0:
        raise ValueError(f"Not enough graph snapshots ({len(snapshots)}) for sequence length {seq_len}.")

    model.train()
    epochs = 10

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for i in range(num_samples):
            seq = [snapshots[i + j].to(device) for j in range(seq_len)]
            target = snapshots[i + seq_len].y.to(device)

            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_samples
        print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights successfully saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()