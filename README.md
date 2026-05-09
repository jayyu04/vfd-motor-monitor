# VFD Motor Monitoring System

A real-time monitoring and anomaly detection system for Variable Frequency Drive (VFD) motors.  
This project combines **physics-informed rule-based analysis** with a **Random Forest classifier** to provide live fault monitoring, risk scoring, and dashboard-based visualization.

> **Live Demo:** [vfd-motor-monitor.streamlit.app](https://vfd-motor-monitor-amlmwsweesyarvhptqxgmz.streamlit.app)

---

## Overview

This project was built as a prototype for **industrial motor condition monitoring**.  
It simulates real-time operating data from a VFD-controlled motor, processes the data through two anomaly detection layers, stores the results in a database, and presents them in an interactive dashboard.

The goal is to demonstrate how a monitoring system can be structured from end to end:

- **data generation / acquisition**
- **feature calculation**
- **rule-based fault scoring**
- **machine learning classification**
- **data persistence**
- **real-time visualization**

---

## Motor Configuration

The simulated motor is based on the following assumptions:

- **Motor Power:** 10 HP (7.46 kW)
- **Supply Voltage:** 380V three-phase AC
- **Pole Count:** 4
- **Rated Current (FLA):** 15A
- **Control Method:** VFD V/f control

---

## System Architecture

```text
dashboard.py          ← Real-time monitoring UI + control panel
    ↓
control.py            ← Unified control entry point
    ↓
simulator.py          ← Motor data simulation
    ↓
device_adapter.py     ← Derived physical quantity calculation
    ↓
main.py               ← Data pipeline orchestration
    ├── rules.py      ← Rule-based anomaly scoring
    └── ml_model.py   ← Random Forest fault classification
    ↓
database.py           ← SQLite data persistence
```

### Data Flow

1. The dashboard issues control actions such as **power on/off** and **operation mode selection**.
2. The simulator generates motor-related sensor data in real time.
3. The device adapter computes derived quantities from raw signals.
4. The main pipeline sends each record to:
   - the **rule-based scoring engine**
   - the **machine learning model**
5. Results are merged into a unified record and written to SQLite.
6. The dashboard reads the stored data and displays:
   - latest operating state
   - trend charts
   - recent alerts
   - recent 100 records

---

## Detection Strategy

This system uses a **dual-layer anomaly detection design**.

### 1) Rule-based Anomaly Scoring

The first layer is a deterministic scoring system based on engineering thresholds and operating logic.  
Each abnormal condition contributes to an overall anomaly score from **0 to 100**.

| Score Range | Risk Level | Suggested Action |
|------------|------------|------------------|
| 0–14       | Normal     | Continue monitoring |
| 15–39      | Warning    | Schedule inspection |
| 40–64      | Danger     | Reduce load / prepare shutdown |
| 65–100     | Critical   | Immediate shutdown |

This layer is designed to be:
- interpretable
- easy to validate
- consistent with industrial alarm logic

### 2) Machine Learning Classification

The second layer uses a **Random Forest classifier** trained on **5,000 simulated records**  
(1,000 samples per fault category).

#### Input Features
- `frequency_hz`
- `current_a`
- `current_ratio`
- `slip_ratio`
- `rpm_est`
- `torque_nm`
- `input_power_kw`
- `current_std_5`

#### Outputs
- predicted fault type
- risk level
- confidence score

This layer is intended to complement rule-based logic by identifying patterns that are less obvious with fixed thresholds alone.

---

## Final Risk Decision

The final system output is determined by taking the **more severe result** between:

- the **rule-based anomaly score**
- the **ML classification result**

This allows the system to combine:

- the **clarity and explainability** of rules
- the **pattern recognition ability** of machine learning

---

## Fault Types

| Fault Type      | Description                                      | Risk Level |
|-----------------|--------------------------------------------------|------------|
| `NORMAL`        | Normal operating condition                       | Normal     |
| `OVERLOAD`      | Overcurrent above expected full-load behavior    | Warning    |
| `BEARING_WEAR`  | Current ripple under relatively stable frequency | Warning    |
| `LOAD_LOSS`     | Load disconnection, belt break, dry pump, etc.   | Danger     |
| `STALL`         | Mechanical jam with current surge                | Critical   |

---

## Dashboard Features

The dashboard is designed as a monitoring interface rather than just a chart viewer.

### Included Features
- **Real-time data stream** (default update interval: 300 ms)
- **Power ON/OFF control**
- **Operation mode switching**
- **Live charts for key variables**
- **Rule score and ML score visualization**
- **Recent alert records**
- **Recent 100 data records**
- **Technical report and user guide integration**

### Monitoring Variables
- Frequency
- RPM
- Current
- Torque
- Rule-based anomaly score
- ML anomaly score

---

## Project Structure

```text
├── dashboard.py          # Streamlit dashboard
├── simulator.py          # Motor sensor data simulator
├── device_adapter.py     # Derived physical quantity calculator
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

- **Language:** Python 3.11+
- **Frontend:** Streamlit + Chart.js
- **Machine Learning:** scikit-learn (Random Forest)
- **Database:** SQLite

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/vfd-motor-monitor.git
cd vfd-motor-monitor

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

> The ML model is trained automatically on first launch and may take around 10–15 seconds.

---

## Design Notes

This project is currently built as a **prototype / demonstrator system**.  
The data source is simulated, but the architecture is intentionally structured so that the simulator can later be replaced with a real hardware interface.

That means the current version is already organized around a realistic monitoring flow:

- signal source
- calculation layer
- analysis layer
- storage layer
- visualization layer

---

## Future Work

Possible next steps for turning this prototype into a more practical industrial system:

- connect to real VFD hardware via **Modbus RTU / Modbus TCP**
- replace simulated input with actual PLC / sensor data
- add **temperature**, **voltage**, and **vibration** sensing
- integrate **FFT-based bearing fault detection**
- migrate SQLite to a cloud or production database
- add alarm notification mechanisms
- support multi-motor monitoring

---

## Disclaimer

This repository is intended for **educational, prototyping, and demonstration purposes**.  
The current data source is simulated and should not be treated as a production-ready industrial protection system without further validation and hardware integration.
