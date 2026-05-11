from __future__ import annotations

import math
from dataclasses import dataclass

from config import (
    EFFICIENCY,
    POLES,
    POWER_FACTOR,
    RATED_CURRENT_A,
)
from data_collector import CollectedRecord


# ---------------------------------------------------------------------------
# 輸出資料結構
# ---------------------------------------------------------------------------
@dataclass
class PhysicsRecord:
    """
    物理計算層的輸出。
    在 CollectedRecord 基礎上加上推算的物理量。

    注意：
        slip_ratio 和 torque_nm 都是理論估算值，不是真實量測值。
        真實場景需要加裝編碼器才能取得精確數值。
    """

    # 來自 CollectedRecord
    timestamp: str
    motor_id: str
    frequency_hz: float
    current_a: float
    voltage_v: float

    # 推算的物理量
    sync_rpm: float  # 同步轉速 RPM（理論值，120 × Hz / 極數）
    slip_ratio: float  # 轉差率（從電流負載比估算）
    torque_nm: float  # 轉矩 N·m（理論值）


# ---------------------------------------------------------------------------
# 物理公式
# ---------------------------------------------------------------------------


def _calc_slip_ratio(current_a: float) -> float:
    """
    用電流負載比估算轉差率。

    正常負載（< 120% FLA）：slip = 0.02 + ratio × 0.05  → 2% ~ 8%
    過載（>= 120% FLA）    ：slip = 0.10 + (ratio-1.2) × 0.15 → 10%+

    說明：
        沒有轉速感測器，所以用電流負載比近似估算。
        真實場景需要編碼器量測真實轉速才能得到準確滑差。
    """
    load_ratio = current_a / RATED_CURRENT_A

    if load_ratio < 1.2:
        slip = 0.02 + load_ratio * 0.05
    else:
        slip = 0.10 + (load_ratio - 1.2) * 0.15

    return round(min(slip, 0.40), 4)  # 上限 40%，避免異常值


def _calc_torque(
    frequency_hz: float,
    current_a: float,
    voltage_v: float,
) -> float:
    """
    推算理論轉矩（N·m）。

    步驟：
        1. 同步轉速 sync_rpm = 120 × Hz / 極數
        2. 三相輸入功率 P_in = √3 × V × I × cosφ
        3. 輸出機械功率 P_out = P_in × η
        4. 角速度 ω = 2π × sync_rpm / 60
        5. 轉矩 T = P_out / ω

    說明：
        使用同步轉速（理論值）計算，不是實際轉速。
        結果為理論轉矩，供趨勢監控參考。
    """
    if frequency_hz <= 0 or current_a <= 0:
        return 0.0

    sync_rpm = 120.0 * frequency_hz / POLES
    if sync_rpm <= 0:
        return 0.0

    p_in = math.sqrt(3) * voltage_v * current_a * POWER_FACTOR
    p_out = p_in * EFFICIENCY
    omega = 2 * math.pi * sync_rpm / 60.0

    torque = p_out / omega
    return round(torque, 2)


# ---------------------------------------------------------------------------
# 對外主入口
# ---------------------------------------------------------------------------


def calculate(record: CollectedRecord) -> PhysicsRecord:
    """
    接收 data_collector 的輸出，推算完整物理量。

    Args:
        record: data_collector.py 傳來的 CollectedRecord

    Returns:
        PhysicsRecord
    """
    sync_rpm = round(120.0 * record.frequency_hz / POLES, 1)
    slip = _calc_slip_ratio(record.current_a)
    torque = _calc_torque(record.frequency_hz, record.current_a, record.voltage_v)

    return PhysicsRecord(
        timestamp=record.timestamp,
        motor_id=record.motor_id,
        frequency_hz=record.frequency_hz,
        current_a=record.current_a,
        voltage_v=record.voltage_v,
        sync_rpm=sync_rpm,
        slip_ratio=slip,
        torque_nm=torque,
    )
