from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from config import MOTOR_ID
from comms import RawSignal


# ---------------------------------------------------------------------------
# 輸出資料結構
# ---------------------------------------------------------------------------
@dataclass
class CollectedRecord:
    """
    資料擷取層的輸出。
    在 RawSignal（Hz / A / V）基礎上加上系統管理資訊。
    """

    timestamp: str
    motor_id: str
    frequency_hz: float
    current_a: float
    voltage_v: float


# ---------------------------------------------------------------------------
# 資料擷取
# ---------------------------------------------------------------------------
def collect(raw: RawSignal) -> CollectedRecord:
    """
    接收 comms 傳來的原始訊號，加上 timestamp 和 motor_id。

    Args:
        raw: comms.py 傳來的 RawSignal

    Returns:
        CollectedRecord
    """
    return CollectedRecord(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        motor_id=MOTOR_ID,
        frequency_hz=raw.frequency_hz,
        current_a=raw.current_a,
        voltage_v=raw.voltage_v,
    )
