# Traxis

**AI-Based Network Attack Forecasting System**
*Smart India Hackathon — Enterprise SOC Command Center*

Traxis is a real-time, spatial-temporal threat intelligence platform. By combining **Graph Neural Networks (GNNs)** with **Gated Recurrent Units (GRUs)**, Traxis doesn't just detect ongoing intrusions — it forecasts multi-stage cyber attacks ahead of time across network host topologies.

---

## Overview

Traditional monitoring reacts after an attack is detected. Traxis models how an attack evolves across a network's host topology over time and predicts its next stage, shifting defense from detection to forecasting.

## Features

- **Predictive Threat Intelligence** — forecasts the next MITRE ATT&CK progression phase (`t+1`) before full-scale impact occurs
- **SOC Command Center UI** — a Streamlit dashboard with a dark-mode theme, glass-panel cards, and live status indicators
- **Network Topology Visualization** — interactive PyVis host graphs highlighting active nodes and traffic flows
- **Explainable AI (XAI)** — real-time feature attribution showing the top telemetry drivers (flow duration, SYN/ACK flags, bytes/sec) behind each prediction
- **Live Telemetry Ingestion** — upload arbitrary network-flow CSV logs for instant graph windowing and threat scoring

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend & UI | Streamlit, PyVis, HTML/CSS |
| Deep Learning Core | PyTorch, PyTorch Geometric, GNN-GRU spatial-temporal architecture |
| Data & Compute | Pandas, NumPy, NetworkX |
| Deployment & Storage | Docker, Render, Git LFS |

## Project Structure

```
traxis/
├── app.py                      # Main Streamlit SOC dashboard
├── Dockerfile                  # Production container configuration for Render
├── requirements.txt            # Pinned dependencies
├── models/
│   └── pretrained_weights.pt   # Pre-trained GNN-GRU weights (LFS tracked)
├── data/
│   └── processed_graphs/
│       └── master_graphs.pt    # Pre-compiled graph snapshots (LFS tracked)
└── src/
    ├── model.py                # NetworkAttackForecaster architecture
    ├── data_loader.py          # Telemetry CSV → graph snapshot pipeline
    ├── train.py                # Training routines
    └── explain.py              # Feature attribution and explainability
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/traxis.git
cd traxis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application locally:

```bash
streamlit run app.py
```

## MITRE ATT&CK Mapping

Traxis classifies threat progression into five operational stages:

| Stage | Description | Color |
|:---:|---|---|
| 0 | Benign / Normal Traffic | `#00F2FE` |
| 1 | Reconnaissance / Initial Access | `#FFB703` |
| 2 | Execution / Persistence | `#FB8500` |
| 3 | Lateral Movement / DoS | `#FF2A6D` |
| 4 | Exfiltration / Impact | `#9D4EDD` |

**This Project is solely build for the SIH 2026 Submission under Problem Statement ```26153: AI based Network Attack Forecasting from Network Traffic Data``` by the team CodeCarnage.**

<div align="center" >
<b>Team Members:</b> Ilsa, Nadeem, Nomaan, Priyanshu, Shrishti, Yash
</div>