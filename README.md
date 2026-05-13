# VFD Motor Monitoring System v4.0

> Real-time VFD motor monitoring and anomaly analysis prototype  
> Rule Confidence + Random Forest · Industrial Monitoring Prototype

**Live Demo:**  
[vfd-motor-monitor.streamlit.app](https://vfd-motor-monitor-amlmwsweesyarvhptqxgmz.streamlit.app)

---

## Overview

**VFD Motor Monitoring System v4.0** is a real-time monitoring and anomaly analysis prototype designed for a **380V / 4-pole / 10HP VFD-controlled motor**.

The project simulates an industrial motor monitoring workflow and combines:

- **modular system architecture**
- **hardware-ready communication abstraction**
- **physics-based feature estimation**
- **rule-based fault interpretation**
- **machine learning classification**
- **database persistence**
- **interactive monitoring dashboard**

Although the current version uses a simulator instead of live hardware input, the overall pipeline is intentionally designed to reflect how a practical industrial monitoring system would be structured.

This project is not just a dashboard demo. It is a **full monitoring prototype** that shows how data can move from signal generation to diagnosis, storage, and visualization in a scalable way.

---

## Why This Project

Industrial monitoring systems are not only about plotting sensor values.  
A meaningful monitoring prototype should answer these questions:

- Where does the data come from?
- How is the system connected to hardware?
- What physical quantities can be inferred from raw signals?
- How should abnormal behavior be detected?
- How can rule-based logic and machine learning complement each other?
- How should the results be stored and presented to users?

This project was built around those questions.

The goal is to demonstrate a monitoring system that is:

- **interpretable**
- **layered**
- **extensible**
- **ready for future hardware integration**

---

## Key Features

- **Real-time monitoring dashboard** built with Streamlit
- **Dual-layer anomaly detection**
  - Rule Confidence for transparent threshold-based reasoning
  - Random Forest for fuzzy boundary classification
- **Startup masking** to suppress false alarms during motor startup
- **Physics layer** for deriving RPM, slip ratio, and torque from raw signals
- **AUTO mode** for automatic fault switching and live ML verification
- **SQLite persistence** for storing full monitoring records
- **Technical report and user guide** integrated into the project
- **Hardware-ready communication design** for future Modbus or real device integration

---

## Supported Fault Modes

The current system supports six operating modes:

| Mode | Description | Typical Behavior |
|------|-------------|------------------|
| `NORMAL` | Normal operation | Stable frequency and current |
| `OVERLOAD` | Overcurrent condition | High current under normal frequency |
| `STALL` | Mechanical jam / locked rotor | Low frequency + very high current |
| `LOAD_LOSS` | Belt break / pump dry-run / load disconnection | Extremely low current |
| `BEARING_WEAR` | Bearing wear tendency | Normal current center value with large ripple |
| `AUTO` | Automatic random fault switching | Used for continuous testing and ML verification |

---

## Motor Specifications

| Parameter | Value |
|----------|-------|
| Rated Power | 10 HP (7.46 kW) |
| Supply Voltage | 380V Three-phase AC |
| Poles | 4 |
| Rated Frequency | 50 Hz |
| Synchronous Speed | 1500 RPM |
| Rated Current (FLA) | 15 A |
| Power Factor | 0.85 |
| Efficiency | 0.88 |
| Control Method | VFD V/f Control |

---

## System Architecture

```text
config.py              → Motor specifications, system parameters, rule thresholds
records.py             → Shared data structure (FullRecord)
VFD_simulator.py       → Hardware simulation layer (outputs only Hz / A / V)
comms.py               → Communication layer (future hardware integration point)
data_collector.py      → Adds timestamp / motor_id
physics.py             → Calculates sync_rpm / slip_ratio / torque_nm
rules.py               → Rule-based fault confidence evaluation
ml_model.py            → Random Forest classifier
main.py                → Pipeline orchestration + startup masking + result merge
control.py             → System state control, single control entry point
database.py            → SQLite persistence layer
dashboard.py           → Streamlit monitoring dashboard

static/
  technical_report.html
  user_guide.html
