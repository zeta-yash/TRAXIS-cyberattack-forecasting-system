import torch
import torch.nn as nn
from pathlib import Path
from model import NetworkAttackForecaster

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed_graphs" / "dev_sample_30k_graphs.pt"
MODEL_SAVE_PATH = BASE_DIR / "models" / "pretrained_weights.pt"

def train():
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: {PROCESSED_DATA_PATH} not found. Run prepare_dataset.py first!")
        return

    # Load graphs
    snapshots = torch.load(PROCESSED_DATA_PATH, weights_only=False)
    print(f"Loaded {len(snapshots)} graph snapshots for training.")

    seq_len = 3  # Use t-2, t-1, t to forecast t+1
    if len(snapshots) <= seq_len:
        print("Not enough snapshots to build temporal sequences.")
        return

    # Automatically set device (Apple Silicon MPS, CUDA, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Initialize Model
    in_channels = snapshots[0].x.shape[1]
    model = NetworkAttackForecaster(in_channels=in_channels, hidden_channels=64, num_classes=5).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    epochs = 15
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        samples_count = 0
        
        # Build sliding temporal windows
        for i in range(len(snapshots) - seq_len):
            input_seq = [snapshots[j].to(device) for j in range(i, i + seq_len)]
            target_label = snapshots[i + seq_len].y.to(device)
            
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target_label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            samples_count += 1
            
        avg_loss = total_loss / samples_count if samples_count > 0 else 0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f}")

    # Save Checkpoint
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel weights saved successfully to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()