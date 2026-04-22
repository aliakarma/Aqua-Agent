# AquaAgent — Implementation Assumptions

This document enumerates all assumptions made during implementation where the
paper did not specify exact values or design choices.  Each assumption is tagged
with an ID (A1–A12) referenced from source code comments and README.md.

---

## A1 — EPANET Network File

**Assumption**: The EPANET `.inp` network file (`data/raw/aquaagent_dma.inp`)
is not distributed with this repository.  When `epyt` is unavailable or the
`.inp` file is absent, `DigitalTwin` falls back to a stochastic mock simulator
(exponential/Gaussian random-walk dynamics with sinusoidal demand cycles).

**Impact**: All currently reported metrics were produced in mock mode.
Mock-mode results are **not** equivalent to EPANET hydraulic results.

**Fix (FIX-04)**: Set `reproducibility.strict_epanet: true` in
`configs/default.yaml` to enforce hard failure when EPANET prerequisites are
missing, preventing silent mock-mode contamination.

**Reference**: `src/env/digital_twin.py`

---

## A2 — Sensor Noise Parameters

**Assumption**: Sensor noise standard deviations are set to
σ_pressure = 0.05 m head and σ_flow = 0.01 L/s, which are consistent with
typical ultrasonic flow meter and pressure transducer specifications.

**Reference**: `configs/network.yaml` (`sensor_noise`), `src/env/sensor_noise.py`

---

## A3 — Leak Profile Weights

**Assumption**: Leak profiles (burst, background, slow-onset) are sampled with
equal probability (33.3% each), matching the IWA infrastructure leakage index
literature.

**Reference**: `configs/network.yaml` (`leak.profile_weights`)

---

## A4 — Feature Dimensionality

**Assumption**: The 12-dimensional per-edge feature vector (d_feat = 12)
encodes: mean flow, std flow, rate-of-change flow, max flow, min flow,
mean pressure, std pressure, rate-of-change pressure, max pressure,
min pressure, demand, and time-of-day (sin encoding).

**Reference**: `src/data/dataset.py` (`_build_features`)

---

## A5 — TCN Dilation Schedule

**Assumption**: TCN dilation schedule is [1, 2, 4, 8] (doubling, following
Bai et al. 2018 "An Empirical Evaluation of Generic Convolutional...").
The paper specifies 4 layers and kernel size 3 but does not list dilations.

**Reference**: `configs/training.yaml` (`ada.tcn.dilations`)

---

## A6 — GAE Lambda

**Assumption**: Generalised Advantage Estimation uses λ = 0.95, which is the
standard value from Schulman et al. 2016 (PPO paper).

**Reference**: `configs/training.yaml` (`mappo.ppo.gae_lambda`)

---

## A7 — Governance Constraint Thresholds

**Assumption**: Consumption cap (C1) is 110% of mean zone demand; fairness
threshold (C2) is a Gini coefficient ≤ 0.3; emergency supply (C3) requires
≥ 40% of normal flow in designated emergency zones.

**Reference**: `configs/governance.yaml`

---

## A8 — Monitoring Agent Warmup

**Assumption**: The Monitoring Agent requires a 30-step warmup window before
producing valid lookback tensors.  During warmup, the ADA returns zero anomaly
flags (no-detection default).

**Reference**: `src/agents/monitoring_agent.py`

---

## A9 — MAPPO CTDE

**Assumption**: MAPPO uses Centralised Training Decentralised Execution (CTDE)
with a shared value function that conditions on the global state, while the
policy (actor) conditions only on local observations.

**Reference**: `src/agents/decision_agent.py`, `src/training/train_mappo.py`

---

## A10 — Dataset Stride

**Assumption**: Training dataset windows are sampled with a stride of 60
time-steps (1 minute at 1 Hz) to reduce dataset size from ~31.5 M samples
to a manageable ~500K windows while preserving temporal diversity.

**Reference**: `configs/training.yaml` (`dataset.stride`)

---

## A11 — Zone-to-Edge Assignment (Round-Robin)

**Assumption**: The Governance Agent assigns edges to demand zones using
round-robin allocation: `zone_id = edge_idx % num_zones`.  This is a crude
approximation; in a real network, zone membership must be derived from the
hydraulic topology.

**Reviewer flag (R1-mn3)**: Acknowledged as a simplification.  Any deployment
on a real network **must** replace this with topology-derived zone membership
obtained from the EPANET `.inp` file or network GIS data.

**Reference**: `src/agents/governance_agent.py` (`_estimate_zone_flows`)

---

## A12 — Passive Data Collection for ADA Pre-Training

**Assumption**: The 365-day simulation dataset used to pre-train the ADA is
collected under passive no-op conditions (the DA takes no corrective actions
during data generation).  This produces a purely observational dataset where
hydraulic anomalies arise only from leak injection dynamics.

**Rationale**: ADA learns to detect leaks from hydraulic signatures without
confounding from control actions.  The DA is then trained separately via MAPPO
on the live environment, where actions affect state.

**Known limitation (R1-mn6)**: The ADA training distribution may not fully
represent post-action hydraulic states, potentially reducing ADA performance
during DA-driven actuation.

**Reference**: `src/data/simulate.py` (line ~129)
