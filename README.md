# ANAFS

**AI-based Network Attack Forecasting System**

ANAFS is a lightweight, graph-native cybersecurity platform that forecasts multi-stage network attack progression before critical system compromise. It combines Graph Neural Networks, GRU-based temporal modeling, and MITRE ATT&CK mapping to turn network flow telemetry into proactive threat forecasts.

---

## Overview

Traditional monitoring reacts after an attack is detected. ANAFS models how an attack evolves across time windows and predicts its next stage, shifting defense from detection to forecasting.

- **Graph-native:** network flows are modeled as graphs, one per time window
- **Temporal:** a GRU tracks attack progression over `k = 5` windows
- **Actionable:** predictions map to MITRE ATT&CK stages with feature-level explanations
- **Lightweight:** ~140 KB model checkpoint

## Architecture

```
Network Flow / CSV Telemetry
            │
            ▼
   Data Cleaning & Graph Build
            │
            ▼
      GCN per Time Window
            │
            ▼
        GRU (k = 5)
            │
            ▼
   Attack Stage Forecasting
            │
      ┌─────┴─────┐
      ▼           ▼
MITRE ATT&CK     XAI
   Mapping    Feature Drivers
```

## Attack Stages

| Stage | Description |
|:-----:|-------------|
| 0 | Normal / Benign Baseline |
| 1 | Reconnaissance / Scanning |
| 2 | Initial Access / Execution |
| 3 | Lateral Movement / DoS |
| 4 | Exfiltration / System Impact |

## Quick Start

ANAFS uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/zeta-yash/anafs.git
cd anafs
uv sync
uv run streamlit run app.py
```

On launch, `app.py` checks for missing datasets and model artifacts and builds lightweight defaults automatically.

### Manual Pipeline (optional)

```bash
uv run python -m src.prepare_dataset   # prepare dataset
uv run python -m src.train             # train model
```

## Dashboard

- **Live CSV upload:** test custom network-flow logs directly
- **Interactive graph:** explore network topology with PyVis
- **SOC baseline tracking:** monitor benign behavior and attack drift
- **Attack forecasting:** predict multi-stage attack progression
- **XAI telemetry drivers:** surface influential features such as SYN counts, packet rates, flags, and flow duration

## Model

| | |
|---|---|
| Architecture | GCN → GRU → 5-stage classifier |
| Temporal context | k = 5 windows |
| Checkpoint size | ~140 KB |
| Dataset | CIC-IDS network-flow telemetry |
| Class imbalance | Dampened inverse class weighting |

## What's New in 2.0

| | v1.0 | v2.0 |
|---|---|---|
| Pipeline | Manual, multi-step | Single-entry, self-healing `app.py` |
| Data | Single offline graph snapshot | Multi-CSV and live upload |
| Temporal context | k = 3 | k = 5 |
| Class imbalance | Standard cross-entropy | Dampened inverse class weighting |
| Baseline | Binary attack detection | Benign baseline with drift scoring |
| Model size | ~1.5 MB | ~140 KB |

## Project Structure

```
anafs/
├── app.py                  # Streamlit dashboard and entry point
├── pyproject.toml
├── models/
│   └── pretrained_weights.pt
├── data/
│   ├── original_samples/
│   ├── cleaned_samples/
│   └── processed_graphs/
└── src/
    ├── data_loader.py
    ├── model.py
    ├── prepare_dataset.py
    ├── train.py
    └── explain.py
```

## Tech Stack

PyTorch Geometric · GCN + GRU · MITRE ATT&CK · SHAP-lite · Streamlit · PyVis · uv