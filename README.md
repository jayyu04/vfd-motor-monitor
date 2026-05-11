# VFD Motor Monitoring System v3.0

> 變頻器馬達異常監測與風險分析系統  
> Rule-based Anomaly Score + Random Forest · 台科大產業碩士專案

---

## 專案簡介

本系統針對 **380V / 4極 / 10HP 變頻器馬達**，在沒有真實硬體的條件下，用正確的工業監控系統架構設計，搭配物理模型和機器學習，建立一個可擴展的異常監測原型系統。

系統採用**雙層判斷架構**：
- **第一層 Rules**：規則式 Anomaly Score（0~100），快速保底，原因透明
- **第二層 ML**：隨機森林 100 棵決策樹投票，處理模糊地帶

---

## 系統架構

```
config.py          → 馬達規格、系統參數、Rules 閾值（集中管理）
records.py         → FullRecord 資料結構（避免循環 import）
VFD_simulator.py   → 硬體模擬層，只輸出 Hz / A / V
comms.py           → 通訊層，換硬體只改這裡（未來接 Modbus）
data_collector.py  → 加 timestamp / motor_id
physics.py         → 推算 sync_rpm / slip_ratio / torque_nm
rules.py           → Anomaly Score 0~100，第一層判斷
ml_model.py        → 隨機森林，6 個特徵，100 棵樹
main.py            → 流水線串接，啟動遮蔽，合併結果
control.py         → 控制層，狀態管理，唯一控制入口
database.py        → SQLite 儲存
dashboard.py       → Streamlit 展示層
static/
  technical_report.html  → 技術報告（10 章節）
  user_guide.html        → 使用說明（8 章節）
```

---

## 資料流

```
VFD_simulator → comms → data_collector → physics
    → main（啟動遮蔽判斷）
        ├── 啟動中（< 5秒）→ 填啟動中/0 → database
        └── 運轉中（≥ 5秒）→ rules + ml → 合併 → database
            → dashboard 顯示
```

---

## 六種模擬工況

| 工況 | 頻率 | 電流 | 等級 |
|------|------|------|------|
| NORMAL | 30~60 Hz | 10.5~14 A | 正常 |
| OVERLOAD | 45~50 Hz | 18.5~22 A | 警告 |
| STALL | 20~35 Hz | 24~30 A | 緊急 |
| LOAD_LOSS | 45~50 Hz | 2~4 A | 危險 |
| BEARING_WEAR | 45~50 Hz（穩定）| 10~13 A（波動大）| 警告 |
| AUTO | 自動隨機切換 | 10~20 秒/次 | — |

---

## ML 設計

- **演算法**：Random Forest（隨機森林）
- **訓練資料**：5000 筆（五種工況各 1000 筆，由 simulator 自動產生）
- **訓練/測試分割**：80% / 20%
- **測試集準確率**：100%
- **6 個輸入特徵**：

```
frequency_hz    → 頻率
current_a       → 電流絕對值
current_ratio   → 電流 / 額定電流（正規化）
slip_ratio      → 轉差率
torque_nm       → 轉矩
current_std_5   → 最近 5 筆電流標準差（BEARING_WEAR 關鍵特徵）
```

---

## 快速開始

### 安裝依賴

```bash
pip install streamlit streamlit-autorefresh pandas scikit-learn numpy
```

### 啟動系統

```bash
streamlit run dashboard.py
```

**第一次啟動**會自動訓練 ML 模型（約 15~20 秒），之後載入 pkl 只需 1~2 秒。

### 重新訓練

刪除 pkl 檔，下次啟動時自動重新訓練：

```bash
rm rf_model.pkl rf_encoder.pkl
```

---

## 資料庫欄位

```
系統資訊：timestamp / motor_id / machine_state
感測器：  frequency_hz / current_a / voltage_v
物理推算：sync_rpm / slip_ratio / torque_nm
Rules：   rule_fault_type / rule_level / rule_score / rule_reasons
ML：      ml_fault_type / ml_level / ml_confidence
綜合：    final_level
```

---

## 設計原則

**低耦合**
- 換真實硬體：只改 `comms.py`
- 換馬達規格：只改 `config.py`
- Rules 和 ML 完全不知道系統狀態，只負責判斷

**啟動遮蔽**
- 開機後 5 秒不呼叫 Rules 和 ML
- 避免把正常的高啟動電流誤判為緊急告警

**雙層互補**
- Rules 快速保底，特徵明顯時立即告警
- ML 處理模糊地帶（NORMAL vs BEARING_WEAR、OVERLOAD vs STALL）
- 綜合等級取兩者較嚴重的

**AUTO 模式**
- 系統自動隨機切換工況，不重複上一個
- Dashboard 即時顯示實際工況 vs ML 判斷 vs 是否正確
- 最直接展示 ML 真實能力的方式

---

## 系統限制（誠實說明）

- 訓練資料來自模擬器，Simulation-to-Real 存在差距
- slip_ratio 和 torque_nm 是從電流估算的理論值，非真實量測
- BEARING_WEAR 是簡化版，真實需要高頻取樣和 FFT
- STALL 的雙條件（電流 + 滑差）因滑差由電流估算，兩者有相關性

---

## 未來擴展

```
換真實硬體      → comms.py 實作 Modbus RTU
換馬達規格      → config.py 更新銘牌規格
重新訓練 ML     → 用真實資料替換模擬訓練資料
升級 BEARING_WEAR → 加裝振動感測器 + FFT 分析
```

---

## 馬達規格

| 參數 | 數值 |
|------|------|
| 額定功率 | 10 HP（7.46 kW）|
| 電源電壓 | 380V 三相交流 |
| 極數 | 4 極 |
| 額定頻率 | 50 Hz |
| 同步轉速 | 1500 RPM |
| 額定電流（FLA）| 15 A |
| 額定功率因數 | 0.85 |
| 效率 | 0.88 |

---

## 版本

**v3.0** — 大幅重構，採用十二模組嚴格分層架構

主要改動：
- 新增 `comms.py`（通訊層）、`data_collector.py`（擷取層）、`physics.py`（物理計算層）、`records.py`（資料結構）
- `VFD_simulator.py` 只輸出 Hz / A / V，移除所有管理資訊
- `fault_type` 不進資料流，不存資料庫
- `machine_state` 改由 `main.py` 根據 `elapsed_sec` 判斷後寫入
- ML 特徵從 8 個精簡為 6 個，移除 `rpm_est` 和 `input_power_kw`
- 新增 AUTO 模式（自動隨機切換工況）
- Dashboard 新增最近工況流程圖和 AUTO 模式即時驗證對比表
