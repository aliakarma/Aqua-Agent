# data/raw — EPANET Network File

## Expected file: `aquaagent_dma.inp`

This directory should contain the EPANET 2.2 network input file for the
AquaAgent DMA (District Metered Area) network with the following topology:

| Parameter       | Value |
|-----------------|-------|
| Nodes           | 261   |
| Pipe segments   | 213   |
| Demand zones    | 48    |
| PRV valves      | 18    |
| Tanks           | 4     |
| Reservoirs      | 2     |

## Current status

The `.inp` file is **not included** in this public release. All reported
results were generated using the stochastic mock simulator built into
`src/env/digital_twin.py`.

## Options for obtaining the network file

### Option A — Generate a synthetic network with epyt
```python
import epyt
# See epyt documentation for network generation utilities.
# The network must match the topology parameters listed above.
```

### Option B — Use a public benchmark network
The L-TOWN or anytown benchmark networks from the Battle of Water Networks
(BWSN) competition are freely available and can be adapted.  See:
  https://www.bathtub.io/benchmarks

### Option C — Provide your own DMA network
Replace this file with any valid EPANET 2.2 `.inp` file and update
`configs/network.yaml` to match your network's node/edge/zone counts.

## Enabling strict EPANET mode

Once `aquaagent_dma.inp` is in place and `epyt` is installed:
```yaml
# configs/default.yaml
reproducibility:
  strict_epanet: true
```
This will raise a `RuntimeError` instead of falling back to mock mode.
