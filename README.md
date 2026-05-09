# VFD Motor Monitoring System

A real-time monitoring and anomaly detection system for Variable Frequency Drive (VFD) controlled motors, combining Rule-based Anomaly Scoring and Random Forest machine learning.

> **Live Demo:** [vfd-motor-monitor.streamlit.app]([https://vfd-motor-monitor.streamlit.app](https://vfd-motor-monitor-amlmwsweesyarvhptqxgmz.streamlit.app))

---

## Overview

This system simulates industrial motor sensor data and analyzes it through two independent layers of anomaly detection, providing real-time risk assessment and alert management through an interactive dashboard.

**Motor Specifications**
- Power: 10 HP (7.46 kW)
- Voltage: 380V Three-phase AC
- Poles: 4
- Rated Current (FLA): 15A
- Control Method: VFD V/f Control

---

## System Architecture

```
dashboard.py          ← Real-time monitoring UI + controls
    ↓
control.py            ← Unified control entry point
    ↓
simulator.py          ← Sensor data simulation (Hz / A / V)
    ↓
device_adapter.py     ← Physical quantity calculation
    ↓
main.py               ← Pipeline orchestration
    ├── rules.py      ← Rule-based Anomaly Score (0~100)
    └── ml_model.py   ← Random Forest classifier
    ↓
database.py           ← SQLite data persistence
```

---

## Anomaly Detection

### Layer 1 — Rule-based Anomaly Score

Physics-based scoring system that calculates an anomaly score (0–100) for each data point. Each condition scores independently and sums to a final score:

| Score | Risk Level | Action |
|-------|-----------|--------|
| 0–14 | Normal | Continue monitoring |
| 15–39 | Warning | Schedule inspection |
| 40–64 | Danger | Reduce load / plan shutdown |
| 65–100 | Critical | Immediate shutdown |

### Layer 2 — Random Forest Classifier

Trained on 5,000 simulated data points (1,000 per fault type) using 8 physical features:

- `frequency_hz`, `current_a`, `current_ratio`, `slip_ratio`
- `rpm_est`, `torque_nm`, `input_power_kw`, `current_std_5`

Outputs fault type, risk level, and confidence score (0–100%).

### Final Risk Level

Takes the **more severe** result between Rule-based and ML outputs — the two layers complement each other.

---

## Fault Types

| Fault Type | Description | Risk Level |
|-----------|-------------|-----------|
| `NORMAL` | Normal operation | Normal |
| `OVERLOAD` | Overcurrent (>110% FLA) | Warning |
| `BEARING_WEAR` | Current ripple at stable frequency | Warning |
| `LOAD_LOSS` | Load disconnection (belt break / dry pump) | Danger |
| `STALL` | Mechanical jam, current surge | Critical |

---

## Features

- **Real-time data stream** — generates sensor data every 300ms
- **Dual-layer detection** — Rules for clear anomalies, ML for fuzzy boundaries
- **Startup masking** — suppresses alerts during motor startup (first 3 seconds)
- **Interactive dashboard** — live charts, alert history, full data table
- **Hardware-ready architecture** — swap `device_adapter.py` to connect real hardware
- **Technical Report & User Guide** — accessible from the dashboard

---

## Project Structure

```
├── dashboard.py          # Streamlit dashboard
├── simulator.py          # Motor sensor simulator
├── device_adapter.py     # Physical quantity calculator
├── control.py            # System control entry point
├── rules.py              # Rule-based anomaly scoring
├── ml_model.py           # Random Forest model
├── main.py               # Pipeline orchestration
├── database.py           # SQLite data storage
├── requirements.txt      # Python dependencies
├── static/
│   ├── technical_report.html
│   └── user_guide.html
└── .streamlit/
    └── config.toml
```

---

## Tech Stack

- **Frontend** — Streamlit + Chart.js
- **ML** — scikit-learn (Random Forest)
- **Database** — SQLite
- **Language** — Python 3.11+

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/vfd-motor-monitor.git
cd vfd-motor-monitor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run dashboard.py
```

The ML model will train automatically on first launch (~15 seconds).

---

## Future Work

- Connect real VFD hardware via Modbus RTU
- Add FFT-based bearing wear detection using vibration sensors
- Upgrade to persistent cloud database for production use
- Add temperature and humidity sensor integration
