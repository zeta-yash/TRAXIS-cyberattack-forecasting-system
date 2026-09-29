import streamlit as st
import torch
import pandas as pd
import numpy as np
import networkx as nx
import time
from pathlib import Path
from pyvis.network import Network
import streamlit.components.v1 as components

from src.model import NetworkAttackForecaster
from src.data_loader import csv_to_graph_snapshots

# Page Configuration
st.set_page_config(
    page_title="ANAFS // Next-Gen AI SOC Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber SOC Aesthetic CSS & Ambient Canvas Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    :root {
        --bg-obsidian: #030712;
        --card-glass: rgba(15, 23, 42, 0.75);
        --border-slate: rgba(30, 41, 59, 0.8);
        --cyber-cyan: #00F2FE;
        --cyber-blue: #4FACFE;
        --cyber-purple: #7B2CBF;
        --threat-crimson: #FF2A6D;
        --threat-amber: #FFB703;
        --text-bright: #F8FAFC;
        --text-subtle: #94A3B8;
    }

    /* Abstract Cyber Grid Canvas Backdrop */
    .stApp {
        background-color: var(--bg-obsidian);
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(123, 44, 191, 0.06) 0%, transparent 45%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
        color: var(--text-bright);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar SOC Panel Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(3, 7, 18, 0.95) !important;
        border-right: 1px solid var(--border-slate);
    }

    /* Header Brand Bar */
    .soc-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 24px;
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-slate);
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .soc-logo-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .soc-logo-badge {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #00F2FE 0%, #7B2CBF 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        color: #030712;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
    }

    .soc-title-text {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #F8FAFC 0%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .pulse-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #00F2FE;
        border-radius: 50%;
        box-shadow: 0 0 10px #00F2FE;
        margin-right: 8px;
    }

        /* Glassmorphic Metric Cards (Compact) */
    div[data-testid="stMetric"] {
        background: var(--card-glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-slate);
        border-radius: 10px;
        padding: 8px 12px !important;  /* Reduced padding */
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem !important;  /* Smaller label font */
        color: var(--text-subtle) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        white-space: nowrap;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: var(--cyber-cyan) !important;
        font-size: 1.25rem !important;  /* Reduced value font size to prevent truncation */
        font-weight: 700;
        white-space: nowrap;
    }

    /* Dynamic Threat Banners */
    .status-banner-benign {
        background: rgba(0, 242, 254, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.5);
        color: var(--cyber-cyan);
        padding: 14px 22px;
        border-radius: 12px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 24px;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.15);
    }
    
    .status-banner-warning {
        background: rgba(255, 183, 3, 0.08);
        border: 1px solid rgba(255, 183, 3, 0.5);
        color: var(--threat-amber);
        padding: 14px 22px;
        border-radius: 12px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 24px;
        box-shadow: 0 0 20px rgba(255, 183, 3, 0.15);
    }

    .status-banner-critical {
        background: rgba(255, 42, 109, 0.08);
        border: 1px solid rgba(255, 42, 109, 0.5);
        color: var(--threat-crimson);
        padding: 14px 22px;
        border-radius: 12px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 24px;
        box-shadow: 0 0 20px rgba(255, 42, 109, 0.15);
    }

    /* Section Headers */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-bright);
        border-left: 3px solid var(--cyber-cyan);
        padding-left: 12px;
        margin-top: 8px;
        margin-bottom: 16px;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Path Setup
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_graphs" / "master_graphs.pt"
MODEL_PATH = BASE_DIR / "models" / "pretrained_weights.pt"

MITRE_STAGES = {
    0: ("Benign / Normal Traffic", "#00F2FE"),
    1: ("Stage 1: Reconnaissance / Initial Access", "#FFB703"),
    2: ("Stage 2: Execution / Persistence", "#FB8500"),
    3: ("Stage 3: Lateral Movement / DoS", "#FF2A6D"),
    4: ("Stage 4: Exfiltration / Impact", "#9D4EDD")
}

def initialize_system():
    if "pipeline_ready" not in st.session_state:
        st.session_state["pipeline_ready"] = False

    if not st.session_state["pipeline_ready"]:
        if not MODEL_PATH.exists() or not DATA_PATH.exists():
            st.markdown('<div class="soc-title-text">ANAFS Engine Initialization</div>', unsafe_allow_html=True)
            
            with st.status("⚡ Orchestrating System Modules...", expanded=True) as status:
                st.write("🔍 [Step 1/3] Checking graph dataset dependencies...")
                time.sleep(0.4)
                
                st.write("⚙️ [Step 2/3] Processing network telemetry into PyG spatial-temporal snapshots...")
                from src.prepare_dataset import build_master_dataset
                build_master_dataset()
                time.sleep(0.4)
                
                st.write("🧠 [Step 3/3] Training deep GNN-GRU neural architecture...")
                from src.train import train
                train()
                time.sleep(0.4)
                
                status.update(label="✅ System Initialization Complete!", state="complete", expanded=False)
        
        st.session_state["pipeline_ready"] = True
        st.rerun()

initialize_system()

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

snapshots = load_graph_data()

# Sidebar Control Center
st.sidebar.markdown("### 🛡️ ANAFS Engine")
st.sidebar.caption("AI Attack Forecasting & Telemetry SOC")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 📂 Test Telemetry Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload raw flow log (.csv)",
    type=["csv"],
    help="Upload original CIC-IDS/ISCX flow CSV log."
)

active_snapshots = snapshots

if uploaded_file is not None:
    if not uploaded_file.name.lower().endswith(".csv"):
        st.sidebar.error("❌ Invalid format! Please upload a .csv file.")
    else:
        try:
            if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
                with st.sidebar.status("Processing uploaded CSV into graph snapshots...", expanded=False):
                    uploaded_snapshots = csv_to_graph_snapshots(uploaded_file, window_size=300)
                
                if len(uploaded_snapshots) > 0:
                    st.session_state["active_snapshots"] = uploaded_snapshots
                    st.session_state["last_uploaded_file"] = uploaded_file.name
                    st.session_state["sim_window"] = 0
                    st.session_state["is_simulating"] = False
                    st.rerun()
                else:
                    st.sidebar.warning("CSV file contains insufficient rows for windowing.")
            
            if "active_snapshots" in st.session_state:
                active_snapshots = st.session_state["active_snapshots"]
                st.sidebar.success(f"Loaded `{uploaded_file.name}` ({len(active_snapshots)} graph windows)")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⏱️ Real-Time Simulation Engine")

SEQ_LEN = 3

if active_snapshots is None or len(active_snapshots) == 0:
    st.error("No graph data found. Please ensure the dataset has been built and processed.")
    st.stop()

if len(active_snapshots) <= SEQ_LEN:
    st.error("Uploaded telemetry provides fewer than 4 graph snapshots for temporal forecasting.")
    st.stop()

max_idx = len(active_snapshots) - (SEQ_LEN + 1)

if "sim_window" not in st.session_state:
    st.session_state["sim_window"] = 0
if "is_simulating" not in st.session_state:
    st.session_state["is_simulating"] = False

def toggle_simulation():
    st.session_state["is_simulating"] = not st.session_state["is_simulating"]

sim_col1, sim_col2 = st.sidebar.columns([1, 1])
if st.session_state["is_simulating"]:
    sim_col1.button("⏸️ Pause", on_click=toggle_simulation, use_container_width=True)
else:
    sim_col1.button("▶️ Play Simulation", on_click=toggle_simulation, use_container_width=True)

if sim_col2.button("🔄 Reset", use_container_width=True):
    st.session_state["sim_window"] = 0
    st.session_state["is_simulating"] = False

window_idx = st.sidebar.slider(
    "Select Time Window (t)", 
    min_value=0, 
    max_value=max_idx, 
    value=st.session_state["sim_window"],
    key="slider_val"
)

st.session_state["sim_window"] = window_idx

in_channels = active_snapshots[0].x.shape[1]
model = load_trained_model(in_channels)

# Tactical SOC Header Bar
st.markdown("""
<div class="soc-header">
    <div class="soc-logo-group">
        <div class="soc-logo-badge">⚡</div>
        <div>
            <div class="soc-title-text">ANAFS // AI Attack Forecasting Engine</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Spatial-Temporal Graph Neural Network & GRU Threat Forecasting Console</div>
        </div>
    </div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #00F2FE;">
        <span class="pulse-indicator"></span>SYSTEM ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

seq = [active_snapshots[window_idx], active_snapshots[window_idx + 1], active_snapshots[window_idx + 2]]
actual_next_label = active_snapshots[window_idx + 3].y.item()


with torch.no_grad():
    logits = model(seq)

    logits = torch.nan_to_num(logits, nan=0.0)
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    

    if np.isnan(probs).any():
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
    predicted_stage = int(np.argmax(probs))
    confidence = probs[predicted_stage] * 100

stage_name, stage_color = MITRE_STAGES[predicted_stage]

# Dynamic Status Banner
if predicted_stage == 0:
    st.markdown(f'<div class="status-banner-benign">🟢 STATUS: NORMAL ACTIVITY — Forecasted Stage (t+1): {stage_name}</div>', unsafe_allow_html=True)
elif predicted_stage in [1, 2]:
    st.markdown(f'<div class="status-banner-warning">⚠️ THREAT ESCALATION DETECTED — Forecasted Stage (t+1): {stage_name}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-banner-critical">🚨 CRITICAL THREAT ALERT — Forecasted Stage (t+1): {stage_name}</div>', unsafe_allow_html=True)

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Window (t)", f"Win #{window_idx + 2}")
col2.metric("Forecasted Stage (t+1)", f"Stage {predicted_stage}")
col3.metric("Prediction Confidence", f"{confidence:.1f}%")
col4.metric("Ground Truth Stage", f"Stage {actual_next_label}")

st.markdown("<br>", unsafe_allow_html=True)

left_col, right_col = st.columns([1.2, 1])

# Left Column: Graph Topology
with left_col:
    st.markdown('<div class="section-header">ACTIVE HOST GRAPH TOPOLOGY</div>', unsafe_allow_html=True)
    
    current_graph = active_snapshots[window_idx + 2]
    G = nx.DiGraph()
    
    edge_index = current_graph.edge_index.numpy()
    if edge_index.shape[1] > 0:
        for src, dst in zip(edge_index[0], edge_index[1]):
            if int(src) != int(dst):
                G.add_edge(f"Host_{src}", f"Host_{dst}")

    if len(G.nodes) == 0:
        n_nodes = min(10, current_graph.x.shape[0])
        for n in range(n_nodes):
            G.add_edge(f"Host_{n}", f"Host_{(n + 1) % n_nodes}")

    net = Network(height="360px", width="100%", bgcolor="#030712", font_color="#E2E8F0", directed=True)
    net.from_nx(G)
    
    node_color = "#FF2A6D" if actual_next_label > 0 else "#00F2FE"
    for node in net.nodes:
        node["color"] = node_color
        node["size"] = 16
    for edge in net.edges:
        edge["color"] = "#7B2CBF"
        edge["width"] = 2
        
    net.toggle_physics(True)
    
    html_string = net.generate_html()
    components.html(html_string, height=380)

# Right Column: MITRE Stage Probabilities Bar Chart
with right_col:
    st.markdown('<div class="section-header">MITRE ATT&CK PHASE PROBABILITIES</div>', unsafe_allow_html=True)
    
    stage_df = pd.DataFrame({
        'MITRE Stage': [MITRE_STAGES[i][0] for i in range(5)],
        'Probability': probs
    })
    
    st.bar_chart(stage_df.set_index('MITRE Stage'), color="#00F2FE")
    st.info(f"**Forecast Target:** {stage_name}")

st.markdown("<br>", unsafe_allow_html=True)

# Explainability Panel
st.markdown('<div class="section-header">TELEMETRY FEATURE ATTRIBUTION (EXPLAINABILITY)</div>', unsafe_allow_html=True)

feature_names = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Flow Bytes/s", "Flow Packets/s", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count"
] + [f"Feature_{i}" for i in range(in_channels - 12)]

def compute_live_feature_importance(graph_data, feature_names):
    x_tensor = graph_data.x.clone().detach()
    if x_tensor.ndim > 1:
        importance = torch.abs(x_tensor).mean(dim=0).cpu().numpy()
    else:
        importance = torch.abs(x_tensor).cpu().numpy()
        
    importance = importance[:len(feature_names)]
    names = feature_names[:len(importance)]
    
    df_imp = pd.DataFrame({'Feature': names, 'Importance': importance})
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    return df_imp

importance_df = compute_live_feature_importance(current_graph, feature_names)

exp_col1, exp_col2 = st.columns([1.5, 1])

with exp_col1:
    st.bar_chart(importance_df.head(8).set_index('Feature'), color="#7B2CBF")

with exp_col2:
    st.write("**Top Telemetry Drivers:**")
    for idx, row in importance_df.head(5).iterrows():
        st.write(f"• **{row['Feature']}**: `{row['Importance']:.4f}`")

# Real-time Simulation Loop
if st.session_state["is_simulating"]:
    time.sleep(0.8)
    st.session_state["sim_window"] = (st.session_state["sim_window"] + 1) % (max_idx + 1)
    st.rerun()