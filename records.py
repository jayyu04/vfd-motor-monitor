from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FullRecord:
    """完整的資料庫紀錄，由 main.py 合併後寫入。"""

    # 系統資訊
    timestamp: str
    motor_id: str
    machine_state: str

    # 感測器原始值
    frequency_hz: float
    current_a: float
    voltage_v: float

    # 物理推算
    sync_rpm: float
    slip_ratio: float
    torque_nm: float

    # Rules 結果
    rule_fault_type: str
    rule_level: str
    rule_score: int
    rule_reasons: str

    # ML 結果
    ml_fault_type: str
    ml_level: str
    ml_confidence: float

    # 綜合結果
    final_level: str
