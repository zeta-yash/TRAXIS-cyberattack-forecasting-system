# create model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class NetworkAttackForecaster(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=5, num_gru_layers=1):
        super(NetworkAttackForecaster, self).__init__()
        
        # 1. Spatial Graph Representation (GCN)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # 2. Temporal Dynamics (GRU)
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=num_gru_layers,
            batch_first=True
        )
        
        # 3. Forecast Classifier Head
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward_snapshot(self, data):
        """Encodes a single spatial graph snapshot into a 1D vector."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Handle single unbatched graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Pool graph nodes into a single graph-level embedding
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding

    def forward(self, sequence_of_graphs):
        """
        Processes a sequence of graph snapshots [t-K, ..., t] 
        and forecasts the attack state for t+1.
        """
        embeddings = []
        for graph in sequence_of_graphs:
            emb = self.forward_snapshot(graph)
            embeddings.append(emb)
            
        # Stack into sequence tensor: [Batch=1, Seq_Len, Hidden_Dim]
        sequence_tensor = torch.stack(embeddings, dim=1)
        
        # Pass through GRU
        gru_out, _ = self.gru(sequence_tensor)
        
        # Take the final temporal state output to predict step t+1
        final_temporal_state = gru_out[:, -1, :]
        
        # Classification logits for MITRE stages (0-4)
        logits = self.fc(final_temporal_state)
        return logits


if __name__ == "__main__":
    # Test model initialization and dummy forward pass
    print("Testing NetworkAttackForecaster model setup...")
    
    # 78 features in CIC-IDS datasets, 64 hidden channels, 5 MITRE stage classes
    model = NetworkAttackForecaster(in_channels=78, hidden_channels=64, num_classes=5)
    print(model)