# VFD Motor Monitoring System v3.0

> 變頻器馬達異常監測與風險分析系統  
> Rule-based Anomaly Score + Random Forest · Industrial Monitoring Prototype

> **Live Demo:** [vfd-motor-monitor.streamlit.app](https://vfd-motor-monitor-amlmwsweesyarvhptqxgmz.streamlit.app)

---

## Overview

**VFD Motor Monitoring System v3.0** is a real-time monitoring and anomaly analysis prototype designed for a **380V / 4-pole / 10HP VFD-controlled motor**.

This project focuses on building a monitoring architecture that is:

- **modular**
- **hardware-extensible**
- **physically interpretable**
- **ready for future integration with real devices**

Although the current version uses simulated data instead of live hardware input, the overall pipeline is intentionally designed to follow a realistic industrial monitoring workflow.

The system uses a **dual-layer detection strategy**:

- **Layer 1 — Rule-based Anomaly Score**  
  Fast, interpretable, threshold-driven monitoring logic

- **Layer 2 — Random Forest Classifier**  
  Handles fuzzy boundaries and pattern-based fault differentiation

---

## Project Goal

The goal of this project is not only to visualize motor data, but to demonstrate a full monitoring pipeline including:

- data source abstraction
- communication layer separation
- physical quantity estimation
- rule-based anomaly scoring
- machine learning classification
- database persistence
- dashboard-based visualization

This makes the project suitable as a **prototype for industrial motor condition monitoring** and a foundation for future hardware-connected versions.

---

## System Architecture

```text
config.py              → Motor specifications, system parameters, rule thresholds
records.py             → Shared data structure (FullRecord)
VFD_simulator.py       → Hardware simulation layer (outputs only Hz / A / V)
comms.py               → Communication layer (future hardware integration point)
data_collector.py      → Adds timestamp / motor_id
physics.py             → Calculates sync_rpm / slip_ratio / torque_nm
rules.py               → Rule-based anomaly scoring (0–100)
ml_model.py            → Random Forest classifier
main.py                → Pipeline orchestration + startup masking + result merge
control.py             → System state control, single control entry point
database.py            → SQLite persistence layer
dashboard.py           → Streamlit monitoring dashboard

static/
  technical_report.html
  user_guide.html
