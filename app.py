import sys
import torch
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in search path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.model import NetworkAttackForecaster
from src.explain import extract_feature_importance
from src.data_loader import csv_to_graph_snapshots
from src.prepare_dataset import build_processed_dataset
from src.train import train_model

# Page Configuration
st.set_page_config(page_title="ANAFS - Network Attack Forecaster", layout="wide")

GRAPH_PATH = BASE_DIR / "data" / "processed_graphs" / "master_graphs.pt"
MODEL_PATH = BASE_DIR / "models" / "pretrained_weights.pt"
RAW_SAMPLES_DIR = BASE_DIR / "data" / "original_samples"

MITRE_STAGES = {
    0: ("Stage 0: Normal / Benign Baseline", "#28a745"),
    1: ("Stage 1: Reconnaissance / Scanning", "#ffc107"),
    2: ("Stage 2: Initial Access / Execution", "#fd7e14"),
    3: ("Stage 3: Lateral Movement / DoS", "#dc3545"),
    4: ("Stage 4: Exfiltration / System Impact", "#6f42c1")
}

def ensure_pipeline_ready():
    """Self-healing pipeline checker."""
    if not GRAPH_PATH.exists():
        with st.spinner("⚡ Initializing master graph dataset from raw telemetry..."):
            build_processed_dataset()
    if not MODEL_PATH.exists():
        with st.spinner("🧠 Training baseline GNN-GRU forecasting model..."):
            train_model()

# Run initialization check
ensure_pipeline_ready()

@st.cache_resource
def load_trained_model(in_channels):
    model = NetworkAttackForecaster(in_channels=in_channels, hidden_channels=64, num_classes=5)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

@st.cache_data
def load_master_snapshots():
    if GRAPH_PATH.exists():
        return torch.load(GRAPH_PATH, weights_only=False)
    return None

# Sidebar Controls
st.sidebar.title("🛡️ ANAFS Engine")
st.sidebar.markdown("**AI-based Network Attack Forecasting**")

data_source = st.sidebar.radio(
    "Data Source Mode:",
    ["Preprocessed Master Graphs", "Upload / Select Raw Flow CSV"]
)

snapshots = None

if data_source == "Preprocessed Master Graphs":
    snapshots = load_master_snapshots()
else:
    raw_files = list(RAW_SAMPLES_DIR.glob("*.csv")) if RAW_SAMPLES_DIR.exists() else []
    selected_file = st.sidebar.selectbox("Select Telemetry Sample:", [f.name for f in raw_files] if raw_files else ["None"])
    
    uploaded_file = st.sidebar.file_uploader("Or Upload Network Telemetry CSV", type=["csv"])
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file, low_memory=False)
        snapshots = csv_to_graph_snapshots(raw_df)
    elif selected_file != "None":
        file_path = RAW_SAMPLES_DIR / selected_file
        raw_df = pd.read_csv(file_path, low_memory=False, encoding_errors='replace')
        snapshots = csv_to_graph_snapshots(raw_df)

if not snapshots or len(snapshots) < 6:
    st.error("Insufficient graph snapshots generated! Need at least 6 time windows.")
    st.stop()

in_channels = snapshots[0].x.shape[1]
model = load_trained_model(in_channels)

# Sequence window selection (k=5 context)
k = 5
max_idx = len(snapshots) - (k + 1)
window_idx = st.sidebar.slider("Select Starting Window (t)", min_value=0, max_value=max_idx, value=0)

# Main Dashboard Title
st.title("Network Baseline Tracking & Multi-Stage Attack Forecaster")
st.caption("Fusing host topology & temporal traffic telemetry to forecast transition risks from benign baselines.")

# Build 5-window context sequence
seq = snapshots[window_idx : window_idx + k]
actual_next_label = snapshots[window_idx + k].y.item()

# Model Inference
with torch.no_grad():
    logits = model(seq)
    # Temperature scaling for smooth, realistic SOC probability distributions
    probs = torch.softmax(logits / 1.2, dim=-1).squeeze().numpy()
    predicted_stage = int(np.argmax(probs))
    confidence = probs[predicted_stage] * 100

stage_name, stage_color = MITRE_STAGES[predicted_stage]
benign_prob = probs[0] * 100
attack_risk_prob = (1.0 - probs[0]) * 100

# Contextual Status Banner
if predicted_stage == 0:
    st.success(
        f"🟢 **BASELINE STABLE:** Network operating within normal parameters. "
        f"Benign Traffic Confidence: **{benign_prob:.1f}%** | Threat Escalation Risk: **{attack_risk_prob:.1f}%**"
    )
elif predicted_stage in [1, 2]:
    st.warning(
        f"⚠️ **ELEVATED DRIFT DETECTED:** Network deviating from benign baseline! "
        f"Forecasted Transition: **{stage_name}** | Attack Progression Probability: **{confidence:.1f}%**"
    )
else:
    st.error(
        f"🚨 **CRITICAL THREAT FORECAST:** Severe baseline anomaly! High confidence multi-stage escalation imminent. "
        f"Target Stage: **{stage_name}** | Forecast Confidence: **{confidence:.1f}%**"
    )

st.markdown("---")

# Metrics Grid: Benign vs Threat Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sequence Window Range", f"t={window_idx} → t={window_idx + k - 1}")
col2.metric("Benign Baseline Score", f"{benign_prob:.1f}%", delta=f"{'-' if predicted_stage != 0 else '+'}{abs(benign_prob-80):.1f}%")
col3.metric("Threat Escalation Risk", f"{attack_risk_prob:.1f}%", delta_color="inverse")
col4.metric("Actual Next Stage (Ground Truth)", f"Stage {actual_next_label}")

st.markdown("---")

# Left Column: Graph Visuals | Right Column: MITRE Forecast Probabilities
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader(f"🌐 Dynamic Host Graph Topology (Window #{window_idx + k - 1})")
    
    current_graph = snapshots[window_idx + k - 1]
    G = nx.DiGraph()
    
    edge_index = current_graph.edge_index.numpy()
    for src, dst in zip(edge_index[0], edge_index[1]):
        G.add_edge(int(src), int(dst))
        
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
    
    if predicted_stage == 0:
        st.info("ℹ️ **SOC Note:** Traffic pattern matches verified benign baseline. No defensive interventions required.")
    else:
        st.error(f"⚠️ **SOC Alert:** Model predicts progression to **{stage_name}**. Inspect host node edges.")

st.markdown("---")

# Feature Attribution / Explainability
st.subheader("🔍 Baseline Anomaly Drivers (Telemetry Attribution)")

feature_names = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Flow Bytes/s", "Flow Packets/s", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count"
] + [f"Feature_{i}" for i in range(in_channels - 12)]

importance_df = extract_feature_importance(snapshots[window_idx + k - 1], feature_names=feature_names)

exp_col1, exp_col2 = st.columns([1.5, 1])

with exp_col1:
    st.bar_chart(importance_df.head(8).set_index('Feature'))

with exp_col2:
    st.write("**Top Telemetry Feature Drivers:**")
    for idx, row in importance_df.head(5).iterrows():
        st.write(f"- **{row['Feature']}**: `{row['Importance']:.4f}`")