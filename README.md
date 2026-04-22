# AquaAgent 🌊

**Agentic AI for Smart Water Distribution: Real-Time Leak Detection and Governance**

Official reproducibility repository for the paper *"AquaAgent: A Governed Multi-Agent AI System for Proactive Leak Detection and Policy-Compliant Control in Smart Water Distribution Networks"*.

---

## 📑 Table of Contents

- [AquaAgent 🌊](#aquaagent-)
  - [📑 Table of Contents](#-table-of-contents)
  - [📌 Overview](#-overview)
  - [🧠 Methodology](#-methodology)
    - [Four-Agent Pipeline](#four-agent-pipeline)
    - [Agent Roles](#agent-roles)
  - [⚙️ Installation](#️-installation)
    - [Prerequisites](#prerequisites)
    - [Linux/macOS](#linuxmacos)
    - [Windows (PowerShell)](#windows-powershell)
  - [🚀 Quick Start (LangChain + Standalone)](#-quick-start-langchain--standalone)
    - [LangChain](#langchain)
    - [Standalone: Full Training Pipeline](#standalone-full-training-pipeline)
    - [Standalone: Quick Smoke Test (CPU)](#standalone-quick-smoke-test-cpu)
    - [Standalone: Python API](#standalone-python-api)
  - [📊 Results](#-results)
    - [Synthetic/Mock Results (Reported)](#syntheticmock-results-reported)
    - [Real EPANET Results](#real-epanet-results)
  - [🔬 Evaluation](#-evaluation)
  - [⚠️ Limitations](#️-limitations)
  - [📂 Repository Structure](#-repository-structure)
  - [🛠️ Development \& Testing](#️-development--testing)
    - [Tests](#tests)
    - [Configuration](#configuration)
  - [📖 Citation](#-citation)
  - [📄 License](#-license)

---

<a id="overview"></a>
## 📌 Overview

AquaAgent is a proactive, policy-aware multi-agent AI system for real-time leak detection and governance in urban water distribution networks (WDNs). The framework is grounded in a **Governed Multi-Agent Markov Decision Process (G-MMDP)** and operates over an EPANET 2.2 digital twin.

The system combines four coordinated agents to monitor network state, detect anomalies, propose control actions, and enforce governance constraints before execution.

---

<a id="methodology"></a>
## 🧠 Methodology

### Four-Agent Pipeline

```
Physical Sensors → Digital Twin (EPANET)
                        │
          ┌─────────────▼──────────────┐
          │       Agent Layer          │
          │                            │
          │  ① Monitoring Agent (MA)   │  Kalman filter + LSTM imputer
          │          ↓                 │  → Edge feature tensors [E × d_feat]
          │  ② Anomaly Detection (ADA) │  TCN (4L, 64ch) → GAT (3-head)
          │          ↓                 │  → Per-edge anomaly scores l̂ ∈ [0,1]
          │  ③ Decision Agent (DA)     │  MAPPO, MLP 512→256→ReLU
          │          ↓                 │  → Candidate action a_t
          │  ④ Governance Agent (GA)   │  c1/c2/c3 constraint check
          │          ↓                 │  → Approved action a* + audit log
          └────────────────────────────┘
                        │
                 Network Actuators
```

### Agent Roles

- **Monitoring Agent (MA)**: Cleans and imputes sensor streams and produces edge-level features.
- **Anomaly Detection Agent (ADA)**: Produces per-edge anomaly probabilities.
- **Decision Agent (DA)**: Proposes control actions using MAPPO.
- **Governance Agent (GA)**: Validates actions against c1/c2/c3 constraints and records audit decisions.

---

<a id="installation"></a>
## ⚙️ Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (recommended) or CPU
- EPANET 2.2 (bundled with `epyt` on Linux/macOS)

### Linux/macOS

```bash
# 1) Clone repository
git clone https://github.com/aliakarma/Aqua-Agent
cd Aqua-Agent

# 2) Create and activate environment
conda create -n aquaagent python=3.10
conda activate aquaagent

# 3) Install PyTorch (adjust CUDA version if needed)
pip install torch==2.2.0 torchvision==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu121

# 4) Install PyTorch Geometric (optional, enables full GAT)
pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric==2.5.0

# 5) Install remaining dependencies
pip install -r requirements.txt

# 6) Install package in editable mode
pip install -e .
```

### Windows (PowerShell)

```powershell
# 1) Clone repository
git clone https://github.com/aliakarma/Aqua-Agent
cd Aqua-Agent

# 2) Create and activate environment
conda create -n aquaagent python=3.10
conda activate aquaagent

# 3) Install PyTorch (adjust CUDA version if needed)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 4) Install PyTorch Geometric (optional, enables full GAT)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric==2.5.0

# 5) Install remaining dependencies
pip install -r requirements.txt

# 6) Install package in editable mode
pip install -e .
```

---

<a id="quick-start"></a>
## 🚀 Quick Start (LangChain + Standalone)

### LangChain

This public repository documents standalone CLI and Python API workflows.

### Standalone: Full Training Pipeline

```bash
# Step 1: Generate 365-day simulation dataset (~31.5M samples)
bash scripts/generate_data.sh 365 42

# Step 2: Pre-train the Anomaly Detection Agent (ADA)
bash scripts/run_train_ada.sh 42 cuda

# Step 3: Train the Decision Agent via MAPPO
bash scripts/run_train_mappo.sh 42 cuda

# Step 4: Evaluate all methods
bash scripts/run_evaluate.sh cuda
```

Windows (PowerShell, equivalent module calls):

```powershell
# Step 1: Generate 365-day simulation dataset (~31.5M samples)
python -m src.data.simulate --config configs/default.yaml --days 365 --seed 42 --output-dir data/generated --splits-dir data/splits

# Step 2: Pre-train the Anomaly Detection Agent (ADA)
python -m src.training.train_ada --config configs/default.yaml --seed 42 --device cuda

# Step 3: Train the Decision Agent via MAPPO
python -m src.training.train_mappo --config configs/default.yaml --seed 42 --device cuda

# Step 4: Evaluate all methods
python -m src.evaluation.evaluate --config configs/default.yaml --device cuda
```

### Standalone: Quick Smoke Test (CPU)

```bash
# Generate only 3 days of data for a fast end-to-end check
python -m src.data.simulate --config configs/default.yaml --days 3 --seed 42

# Train ADA
python -m src.training.train_ada --config configs/default.yaml --device cpu

# Train MAPPO for a short run
# (set mappo.ppo.total_steps = 500 in configs/training.yaml first)
python -m src.training.train_mappo --config configs/default.yaml --device cpu
```

### Standalone: Python API

```python
import numpy as np
from src.data.simulate import load_config
from src.agents.anomaly_agent import AnomalyDetectionAgent
from src.agents.decision_agent import DecisionAgent, build_obs_vector
from src.agents.governance_agent import GovernanceAgent
from src.env.digital_twin import DigitalTwin

cfg = load_config("configs/default.yaml")
dt = DigitalTwin(cfg, seed=42)
state = dt.reset()

ada = AnomalyDetectionAgent(cfg, device="cpu")
ada.load("checkpoints/ada_best.pt")
ada.freeze()

gov = GovernanceAgent(
    cfg,
    num_edges=213,
    num_nodes=261,
    num_zones=48,
    num_valves=18,
)

obs_dim = 213 + 261 + 261 + 213 + 5
da = DecisionAgent(obs_dim=obs_dim, cfg=cfg)
da.load("checkpoints/da_best.pt")

flags = np.zeros(213, dtype=np.float32)
obs = build_obs_vector(state, flags)
action, logp, value = da.act(obs)
exec_action, overridden = gov.validate(action, state)
state, done = dt.step(exec_action)

print(f"Override: {overridden} | PCR: {gov.get_policy_compliance_rate():.3f}")
```

---

<a id="results"></a>
## 📊 Results

### Synthetic/Mock Results (Reported)

| Method | F1 | AUC-ROC | WLR% | PCR% | RT (s) |
|---|---|---|---|---|---|
| **AquaAgent** | **0.934** | **0.971** | **31.2%** | **90.9%** | **48.3** |
| B1 Threshold | 0.643 | 0.712 | 4.1% | — | 312.7 |
| B2 LSTM | 0.781 | 0.834 | — | — | — |
| B3 No-Gov | 0.921 | 0.963 | 28.4% | 62.3% | 51.1 |
| B4 Rules | 0.712 | 0.768 | 11.3% | 100% | 95.4 |

### Real EPANET Results

The public release does not include `data/raw/aquaagent_dma.inp`; therefore, real EPANET-mode results are not reported in this README. The table above corresponds to mock-simulation mode and is not directly comparable to full EPANET hydraulic results.

---

<a id="evaluation"></a>
## 🔬 Evaluation

Run the evaluation stage after training:

```bash
bash scripts/run_evaluate.sh cuda
```

Primary metrics used in the repository include F1, AUC-ROC, WLR%, PCR%, and runtime (RT).

For experiment tracking:

```bash
tensorboard --logdir logs/
```

Available dashboards:

- `train/loss`, `train/f1`, `train/auc` — ADA pre-training
- `mappo/ep_return`, `mappo/pcr`, `mappo/leak_loss` — MAPPO rollout
- `mappo/policy_loss`, `mappo/entropy`, `mappo/clip_frac` — PPO update
- `eval/mean_return`, `eval/wlr`, `eval/pcr` — Evaluation checkpoints

---

<a id="limitations"></a>
## ⚠️ Limitations

- All currently reported results are from **mock-simulation mode**.
- Default behavior is `reproducibility.strict_epanet: false` in `configs/default.yaml`.
- Enabling strict EPANET mode (`strict_epanet: true`) requires `.inp` + `epyt` and raises `RuntimeError` if prerequisites are missing.
- See `data/raw/README.md` for network file instructions.
- Key assumptions are documented in `ASSUMPTIONS.md` (A1, A2, A4, A5, A9, A11, A12).

---

<a id="repository-structure"></a>
## 📂 Repository Structure

```text
.
├── configs/
│   ├── default.yaml
│   ├── governance.yaml
│   ├── network.yaml
│   └── training.yaml
├── data/
│   ├── raw/
│   ├── generated/   # generated
│   └── splits/      # generated
├── scripts/
│   ├── generate_data.sh
│   ├── run_train_ada.sh
│   ├── run_train_mappo.sh
│   └── run_evaluate.sh
├── src/
│   ├── agents/
│   ├── baselines/
│   ├── data/
│   ├── env/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── tests/
├── ASSUMPTIONS.md
├── requirements.txt
├── setup.py
└── README.md
```

---

<a id="development-and-testing"></a>
## 🛠️ Development & Testing

### Tests

```bash
# Run full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test module
pytest tests/test_metrics.py -v
```

### Configuration

All experiment settings are controlled through YAML files in `configs/`.

| Config | Controls |
|---|---|
| `default.yaml` | Paths, seeds, device, logging |
| `network.yaml` | EPANET network, sensor noise, leak injection |
| `training.yaml` | ADA (TCN/GAT), MAPPO (PPO hyperparameters), dataset splits |
| `governance.yaml` | c1/c2/c3 constraint parameters, audit logging |

Multi-run example:

```bash
for seed in 42 43 44 45 46; do
  bash scripts/run_train_ada.sh $seed cuda
  bash scripts/run_train_mappo.sh $seed cuda
done
```

---

<a id="citation"></a>
## 📖 Citation

```bibtex
@article{aquaagent2024,
  title   = {AquaAgent: A Governed Multi-Agent AI System for Proactive Leak
             Detection and Policy-Compliant Control in Smart Water Distribution Networks},
  year    = {2024},
}
```

---

<a id="license"></a>
## 📄 License

MIT License. See `LICENSE` for details.
