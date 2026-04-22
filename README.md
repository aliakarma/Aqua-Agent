# AquaAgent 🌊

**Agentic AI for Smart Water Distribution: Real-Time Leak Detection and Governance**

> Official reproducibility repository for the paper *"AquaAgent: A Governed Multi-Agent AI System for Proactive Leak Detection and Policy-Compliant Control in Smart Water Distribution Networks"*

---

## Overview

AquaAgent is a proactive, policy-aware multi-agent AI system for real-time leak detection and governance in urban water distribution networks (WDNs). The framework is grounded in a **Governed Multi-Agent Markov Decision Process (G-MMDP)** and operates over a high-fidelity EPANET 2.2 digital twin.

### Four-Agent Architecture

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

### Key Results (Paper Table 1)

| Method | F1 | AUC-ROC | WLR% | PCR% | RT (s) |
|---|---|---|---|---|---|
| **AquaAgent** | **0.934** | **0.971** | **31.2%** | **90.9%** | **48.3** |
| B1 Threshold | 0.643 | 0.712 | 4.1% | — | 312.7 |
| B2 LSTM | 0.781 | 0.834 | — | — | — |
| B3 No-Gov | 0.921 | 0.963 | 28.4% | 62.3% | 51.1 |
| B4 Rules | 0.712 | 0.768 | 11.3% | 100% | 95.4 |

---

## Repository Structure

```
aquaagent/
├── configs/                  # YAML experiment configs
│   ├── default.yaml          # Master config
│   ├── network.yaml          # EPANET network parameters
│   ├── training.yaml         # ADA + MAPPO hyperparameters
│   └── governance.yaml       # Governance constraint definitions
├── data/
│   ├── raw/                  # EPANET .inp network file
│   ├── generated/            # Simulated HDF5 dataset (generated)
│   └── splits/               # Train/val/test index arrays (generated)
├── src/
│   ├── env/                  # Digital twin, leak injection, sensor noise
│   ├── agents/               # MA, ADA, DA, GA implementations
│   ├── models/               # TCN, GAT, PPO-MLP architectures
│   ├── training/             # train_ada.py, train_mappo.py
│   ├── evaluation/           # evaluate.py, metrics.py
│   ├── baselines/            # B1–B4 baseline implementations
│   ├── data/                 # Dataset loader, simulation runner
│   └── utils/                # Seed, logging, audit ledger, graph utils
├── scripts/                  # Shell scripts for each pipeline stage
├── notebooks/                # Jupyter analysis notebooks
├── tests/                    # pytest unit tests (47 tests)
├── checkpoints/              # Saved model weights (generated)
├── logs/                     # TensorBoard logs + audit ledger (generated)
├── figures/                  # Generated plots
├── requirements.txt
├── setup.py
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (recommended) or CPU
- EPANET 2.2 (bundled with `epyt` on Linux/macOS)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/aquaagent.git
cd aquaagent

# 2. Create and activate conda environment
conda create -n aquaagent python=3.10
conda activate aquaagent

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Install PyTorch Geometric (optional, enables full GAT)
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric==2.5.0

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Install AquaAgent as a package (editable)
pip install -e .
```

---

## Usage

### Full Training Pipeline

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

### Quick Smoke Test (CPU, 1 day of data)

```bash
# Generate only 3 days of data for a fast end-to-end check
python -m src.data.simulate --config configs/default.yaml --days 3 --seed 42

# Train ADA for 5 epochs
python -m src.training.train_ada --config configs/default.yaml --device cpu

# Train MAPPO for 500 steps
# (edit configs/training.yaml: mappo.ppo.total_steps = 500 first)
python -m src.training.train_mappo --config configs/default.yaml --device cpu
```

### Python API

```python
from src.data.simulate import load_config
from src.agents.anomaly_agent import AnomalyDetectionAgent
from src.agents.decision_agent import DecisionAgent, build_obs_vector
from src.agents.governance_agent import GovernanceAgent
from src.env.digital_twin import DigitalTwin

cfg = load_config("configs/default.yaml")
dt  = DigitalTwin(cfg, seed=42)
state = dt.reset()

ada = AnomalyDetectionAgent(cfg, device="cpu")
ada.load("checkpoints/ada_best.pt")
ada.freeze()

gov = GovernanceAgent(cfg, num_edges=213, num_nodes=261,
                      num_zones=48, num_valves=18)

obs_dim = 213 + 261 + 261 + 213 + 5   # flows + pres + dem + flags + exog
da = DecisionAgent(obs_dim=obs_dim, cfg=cfg)
da.load("checkpoints/da_best.pt")

# One decision cycle
import torch, numpy as np
flags = np.zeros(213, dtype=np.float32)
obs   = build_obs_vector(state, flags)
action, logp, value = da.act(obs)
exec_action, overridden = gov.validate(action, state)
state, done = dt.step(exec_action)
print(f"Override: {overridden} | PCR: {gov.get_policy_compliance_rate():.3f}")
```

---

## Running Tests

```bash
# Run full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test module
pytest tests/test_metrics.py -v
```

---

## Configuration

All hyperparameters are controlled via YAML configs in `configs/`:

| Config | Controls |
|---|---|
| `default.yaml` | Paths, seeds, device, logging |
| `network.yaml` | EPANET network, sensor noise, leak injection |
| `training.yaml` | ADA (TCN/GAT), MAPPO (PPO hyperparams), dataset splits |
| `governance.yaml` | c1/c2/c3 constraint parameters, audit logging |

To run with a different seed for multi-run evaluation:
```bash
for seed in 42 43 44 45 46; do
    bash scripts/run_train_ada.sh $seed cuda
    bash scripts/run_train_mappo.sh $seed cuda
done
```

---

## Monitoring

Launch TensorBoard to track training progress:
```bash
tensorboard --logdir logs/
```

Available dashboards:
- `train/loss`, `train/f1`, `train/auc` — ADA pre-training
- `mappo/ep_return`, `mappo/pcr`, `mappo/leak_loss` — MAPPO rollout
- `mappo/policy_loss`, `mappo/entropy`, `mappo/clip_frac` — PPO update
- `eval/mean_return`, `eval/wlr`, `eval/pcr` — Evaluation checkpoints

---

## Reproducibility Notice

> ⚠️ **Important**: All results reported in `logs/results.json` and in the paper
> Table 1 were produced in **mock-simulation mode** (stochastic random-walk dynamics),
> because the EPANET `.inp` network file (`data/raw/aquaagent_dma.inp`) is not
> included in this public release.
>
> - **Default (mock mode)**: `reproducibility.strict_epanet: false` in `configs/default.yaml`
> - **EPANET mode** (requires `.inp` + `epyt`): set `strict_epanet: true` — raises
>   `RuntimeError` if prerequisites are missing.
>
> See `data/raw/README.md` for instructions on obtaining or generating the network file.
> Mock-mode metrics are **not** directly comparable to full EPANET hydraulic results.

---

## Assumptions

Full list in `ASSUMPTIONS.md`. Key assumptions:

- **A1**: Falls back to stochastic mock simulation when `epyt` or `.inp` is unavailable.  
  All currently reported results use mock mode (see Reproducibility Notice above).
- **A2**: Sensor noise σ_pressure=0.05 m, σ_flow=0.01 L/s.
- **A4**: d_feat=12 features: mean/std/roc/max/min flow & pressure, demand, time-sin.
- **A5**: TCN dilation schedule [1, 2, 4, 8] (Bai et al. 2018).
- **A9**: MAPPO uses Centralised Training Decentralised Execution (CTDE).
- **A11**: Zone-to-edge assignment uses round-robin allocation (must be replaced with
  topology-derived zone membership for real deployments).
- **A12**: ADA training data uses passive no-op conditions (observational dataset).

---

## Citation

## Citation

```bibtex
@article{aquaagent2024,
  title   = {AquaAgent: A Governed Multi-Agent AI System for Proactive Leak
             Detection and Policy-Compliant Control in Smart Water Distribution Networks},
  year    = {2024},
}
```

---

## License

MIT License. See `LICENSE` for details.
