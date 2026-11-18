# Autonomous Thermal and Radiation Mitigation in Sustainable Data Centers using ML and Agentic AI

**B.Tech Minor Project (2026–27)**  
**Department of Electronics and Communication Engineering (AI)**  
**Indira Gandhi Delhi Technical University for Women (IGDTUW), Delhi**  
**Author:** Kritika Raj

## Overview
This project implements a proactive, closed-loop thermal controller for a 12-rack AI data center row. Instead of reacting after hotspots form, a Gradient-Boosted Regression Tree (GBRT) model forecasts rack-level temperature 10 minutes ahead. A coordinated multi-agent layer (Workload Agent + Precision Cooling Agent + Supervisor) acts on the forecast, bounded by a hard Safety & SLA Shield enforcing ASHRAE Class A1 thermal limits ($27\,^\circ\text{C}$ soft target, $32\,^\circ\text{C}$ hard limit).

## Key Results (5 Unseen Evaluation Days)
- **10-min Forecaster:** $\text{MAE} = 0.18\,^\circ\text{C}$, $\text{RMSE} = 0.30\,^\circ\text{C}$, $R^2 = 0.95$ (54% lower error than persistence).
- **Vs. Static Cooling:** Reduced rack-minutes above $27\,^\circ\text{C}$ from **22.01%** to **2.85%** (87% relative reduction) and peak temperature from $30.16\,^\circ\text{C}$ to $27.26\,^\circ\text{C}$ at a $7.7\%$ cooling-energy increase.
- **Vs. Reactive Threshold Control:** Lowered mean hottest-rack temperature by **$0.42\,^\circ\text{C}$** at identical total cooling energy ($913.0\text{ kWh/day}$).
- **Safety:** **0** breaches of the $32\,^\circ\text{C}$ hard limit.

## How to Run Locally

### 1. Run the Simulation & Reproduce Report Figures/Tables
```bash
pip install -r requirements.txt
python sim.py
