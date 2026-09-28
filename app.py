import streamlit as st
import torch
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from pyvis.network import Network
import streamlit.components.v1 as components

from src.model import NetworkAttackForecaster
from src.explain import extract_feature_importance

# Page Config
st.set_page_config(page_title="ANAFS - Network Attack Forecaster", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_graphs" / "dev_sample_30k_graphs.pt"
MODEL_PATH = BASE_DIR / "models" / "pretrained_weights.pt"

MITRE_STAGES = {
    0: ("Benign / Normal Traffic", "#28a745"),
    1: ("Stage 1: Reconnaissance / Initial Access", "#ffc107"),
    2: ("Stage 2: Execution / Persistence", "#fd7e14"),
    3: ("Stage 3: Lateral Movement / DoS", "#dc3545"),
    4: ("Stage 4: Exfiltration / Impact", "#6f42c1")
}

@st.cache_resource
def load_trained_model(in_channels):
    model = NetworkAttackForecaster(in_channels=in_channels, hidden_channels=64, num_classes=5)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

@st.cache_data
def load_graph_data():
    if DATA_PATH.exists():
        return torch.load(DATA_PATH, weights_only=False)
    return None

# Sidebar Setup
st.sidebar.title("🛡️ ANAFS Engine")
st.sidebar.markdown("**AI-based Network Attack Forecasting**")

snapshots = load_graph_data()

if snapshots is None:
    st.error("No processed graph data found! Run `uv run python src/prepare_dataset.py` first.")
    st.stop()

in_channels = snapshots[0].x.shape[1]
model = load_trained_model(in_channels)

# Time Window Selection
max_idx = len(snapshots) - 4
window_idx = st.sidebar.slider("Select Time Window (t)", min_value=0, max_value=max_idx, value=0)

# Dashboard Title
st.title("Network State Transition & Attack Forecasting")
st.caption("Fusing flow telemetry into dynamic host-graphs to forecast multi-step attack progression.")

# Build Input Sequence [t-2, t-1, t]
seq = [snapshots[window_idx], snapshots[window_idx + 1], snapshots[window_idx + 2]]
actual_next_label = snapshots[window_idx + 3].y.item()

# Model Inference
with torch.no_grad():
    logits = model(seq)
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    predicted_stage = int(np.argmax(probs))
    confidence = probs[predicted_stage] * 100

stage_name, stage_color = MITRE_STAGES[predicted_stage]

# Dynamic Alert Callout Banner
if predicted_stage == 0:
    st.success(f"🟢 **STATUS:** Normal Network Activity — Next Window Forecast: {stage_name}")
elif predicted_stage in [1, 2]:
    st.warning(f"⚠️ **ATTACK WARNING:** Early Progression Detected — Next Window Forecast: {stage_name}")
else:
    st.error(f"🚨 **CRITICAL ALERT:** High-Impact Attack Imminent — Next Window Forecast: {stage_name}")

st.markdown("---")

# Metric Summary Panel
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Window (t)", f"Window #{window_idx + 2}")
col2.metric("Forecasted Stage (t+1)", f"Stage {predicted_stage}")
col3.metric("Prediction Confidence", f"{confidence:.1f}%")
col4.metric("Actual Stage (Ground Truth)", f"Stage {actual_next_label}")

st.markdown("---")

# Visualization Section: Interactive Network Topology + Probability Breakdown
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader(f"🌐 Interactive Host Graph Topology (Window #{window_idx + 2})")
    
    current_graph = snapshots[window_idx + 2]
    G = nx.DiGraph()
    
    edge_index = current_graph.edge_index.numpy()
    for src, dst in zip(edge_index[0], edge_index[1]):
        G.add_edge(int(src), int(dst))
        
    # Interactive PyVis Network Construction
    net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    net.from_nx(G)
    net.toggle_physics(True)
    
    html_path = BASE_DIR / "temp_graph.html"
    net.save_graph(str(html_path))
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=420)

with right_col:
    st.subheader("🔮 MITRE ATT&CK Forecast Probabilities")
    
    stage_df = pd.DataFrame({
        'MITRE Stage': [MITRE_STAGES[i][0] for i in range(5)],
        'Probability': probs
    })
    
    st.bar_chart(stage_df.set_index('MITRE Stage'))
    st.info(f"**Target State:** {stage_name}")

st.markdown("---")

# Bottom Section: Feature Explainability Panel
st.subheader("🔍 Prediction Attribution & Telemetry Drivers")

feature_names = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Flow Bytes/s", "Flow Packets/s", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count"
] + [f"Feature_{i}" for i in range(in_channels - 12)]

importance_df = extract_feature_importance(snapshots[window_idx + 2], feature_names=feature_names)

exp_col1, exp_col2 = st.columns([1.5, 1])

with exp_col1:
    st.bar_chart(importance_df.head(8).set_index('Feature'))

with exp_col2:
    st.write("**Top Telemetry Drivers:**")
    for idx, row in importance_df.head(5).iterrows():
        st.write(f"- **{row['Feature']}**: `{row['Importance']:.4f}`")