from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict

from simulator import FaultType, MachineState, PowerState, RawSensorRecord

# ---------------------------------------------------------------------------
# 馬達固定規格（與 simulator 一致）
# ---------------------------------------------------------------------------
POLES = 4
VOLTAGE_V = 380.0
POWER_FACTOR = 0.85  # 額定功率因數 cos φ
EFFICIENCY = 0.88  # 額定效率 η
RATED_CURRENT = 15.0  # FLA


# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------
@dataclass
class AdaptedRecord:
    # ── 原始感測值（直接從 RawSensorRecord 帶過來）──
    timestamp: str
    motor_id: str
    power_state: PowerState
    machine_state: MachineState
    fault_type: FaultType

    frequency_hz: float
    current_a: float
    voltage_v: float

    # ── 推算值（device_adapter 計算）──
    sync_rpm: float  # 同步轉速 = 120f / P
    slip_ratio: float  # 轉差率（由電流負載比推算）
    rpm_est: float  # 實際轉速 = sync_rpm × (1 - slip)
    power_factor: float  # 功率因數（依負載比動態估算）
    input_power_kw: float  # 三相輸入電功率 = √3 × V × I × cosφ
    output_power_kw: float  # 輸出機械功率 = input × η
    torque_nm: float  # 轉矩 = P_out / ω


# ---------------------------------------------------------------------------
# 物理推算函式
# ---------------------------------------------------------------------------


def _calc_sync_rpm(frequency_hz: float, poles: int = POLES) -> float:
    """同步轉速 = 120f / P"""
    return round((120.0 * frequency_hz) / poles, 2)


def _calc_slip(current_a: float, machine_state: MachineState) -> float:
    """
    轉差率由電流負載比推算（簡化模型）：
    - 正常運轉：負載越高，轉差越大（範圍 2%~8%）
    - 啟動期間：轉差率較大（15%~35%）
    - 關機：0
    """
    if machine_state == "OFF":
        return 0.0
    if machine_state == "STARTUP":
        # 變頻器啟動電流約 1.5~2 倍額定，滑差較大但不像直接啟動那麼極端
        load_ratio = min(current_a / (RATED_CURRENT * 2.0), 1.0)
        return round(0.15 - load_ratio * 0.08, 4)  # 7%~15%

    # RUNNING
    load_ratio = min(current_a / RATED_CURRENT, 2.0)
    # 正常負載：2%~8%，超過 120% FLA 時滑差明顯增大（> 10%）
    if load_ratio > 1.2:
        slip = 0.10 + (load_ratio - 1.2) * 0.15  # 10%~25%
    else:
        slip = 0.02 + load_ratio * 0.05  # 2%~8%
    return round(min(slip, 0.25), 4)


def _calc_power_factor(current_a: float, machine_state: MachineState) -> float:
    """
    功率因數依負載比動態估算：
    - 輕載時 pf 偏低（0.60 左右）
    - 滿載時接近額定 0.85
    - 啟動時 pf 較低（0.45~0.65）
    """
    if machine_state == "OFF":
        return 0.0
    if machine_state == "STARTUP":
        return round(min(0.45 + (current_a / (RATED_CURRENT * 5.0)) * 0.20, 0.65), 3)

    load_ratio = min(current_a / RATED_CURRENT, 1.5)
    pf = 0.60 + load_ratio * 0.25
    return round(min(pf, POWER_FACTOR), 3)


def _calc_input_power(current_a: float, voltage_v: float, power_factor: float) -> float:
    """三相輸入電功率 P_in = √3 × V × I × cosφ（kW）"""
    return round(math.sqrt(3) * voltage_v * current_a * power_factor / 1000.0, 4)


def _calc_output_power(input_kw: float) -> float:
    """輸出機械功率 P_out = P_in × η（kW）"""
    return round(input_kw * EFFICIENCY, 4)


# 額定轉矩 = 7460W / (2π × 1450/60) ≈ 49 N·m
RATED_TORQUE_NM = 49.0
# 啟動轉矩上限：額定的 2.5 倍（變頻器控制下的合理最大值）
MAX_STARTUP_TORQUE_NM = RATED_TORQUE_NM * 2.5  # ≈ 122 N·m


def _calc_torque(output_kw: float, rpm: float, machine_state: str = "RUNNING") -> float:
    """
    轉矩 T = P_out / ω
    ω = 2π × RPM / 60（rad/s）
    rpm = 0 時回傳 0 避免除以零。
    啟動期間加上限，避免低轉速時 T = P/ω 爆大。
    """
    if rpm <= 0:
        return 0.0
    omega = (2.0 * math.pi * rpm) / 60.0
    torque = round((output_kw * 1000.0) / omega, 2)

    if machine_state == "STARTUP":
        torque = min(torque, MAX_STARTUP_TORQUE_NM)

    return torque


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def adapt(raw: RawSensorRecord) -> AdaptedRecord:
    """
    接收 simulator（或未來真實硬體）的 RawSensorRecord，
    推算完整物理量，回傳 AdaptedRecord 給 main.py。
    """
    # 關機時全部歸零
    if raw.power_state == "OFF":
        return AdaptedRecord(
            timestamp=raw.timestamp,
            motor_id=raw.motor_id,
            power_state=raw.power_state,
            machine_state=raw.machine_state,
            fault_type=raw.fault_type,
            frequency_hz=0.0,
            current_a=0.0,
            voltage_v=0.0,
            sync_rpm=0.0,
            slip_ratio=0.0,
            rpm_est=0.0,
            power_factor=0.0,
            input_power_kw=0.0,
            output_power_kw=0.0,
            torque_nm=0.0,
        )

    sync_rpm = _calc_sync_rpm(raw.frequency_hz)
    slip = _calc_slip(raw.current_a, raw.machine_state)
    rpm_est = round(sync_rpm * (1.0 - slip), 2)
    pf = _calc_power_factor(raw.current_a, raw.machine_state)
    input_kw = _calc_input_power(raw.current_a, raw.voltage_v, pf)
    output_kw = _calc_output_power(input_kw)
    torque_nm = _calc_torque(output_kw, rpm_est, raw.machine_state)

    return AdaptedRecord(
        timestamp=raw.timestamp,
        motor_id=raw.motor_id,
        power_state=raw.power_state,
        machine_state=raw.machine_state,
        fault_type=raw.fault_type,
        frequency_hz=raw.frequency_hz,
        current_a=raw.current_a,
        voltage_v=raw.voltage_v,
        sync_rpm=sync_rpm,
        slip_ratio=slip,
        rpm_est=rpm_est,
        power_factor=pf,
        input_power_kw=input_kw,
        output_power_kw=output_kw,
        torque_nm=torque_nm,
    )


def adapted_to_dict(record: AdaptedRecord) -> Dict[str, object]:
    return asdict(record)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo() -> None:
    from simulator import generate_sensor_record

    test_cases = [
        ("OFF", "NORMAL", 0.0),
        ("ON", "NORMAL", 0.0),  # 啟動初期
        ("ON", "NORMAL", 1.5),  # 啟動中段
        ("ON", "NORMAL", 4.0),  # 正常運轉
        ("ON", "OVERLOAD", 4.0),
        ("ON", "STALL", 4.0),
        ("ON", "LOAD_LOSS", 4.0),
        ("ON", "BEARING_WEAR", 4.0),
    ]

    for power_state, fault_type, elapsed in test_cases:
        raw = generate_sensor_record(power_state, fault_type, elapsed)
        rec = adapt(raw)

        print("=" * 70)
        print(f"fault_type={fault_type:<14} machine_state={rec.machine_state}")
        print(
            f"  Hz={rec.frequency_hz:6.2f}  "
            f"A={rec.current_a:6.2f}  "
            f"V={rec.voltage_v:6.1f}"
        )
        print(
            f"  sync_rpm={rec.sync_rpm:7.1f}  "
            f"rpm_est={rec.rpm_est:7.1f}  "
            f"slip={rec.slip_ratio:.3f}"
        )
        print(
            f"  pf={rec.power_factor:.3f}  "
            f"P_in={rec.input_power_kw:.3f}kW  "
            f"P_out={rec.output_power_kw:.3f}kW  "
            f"T={rec.torque_nm:.2f}N·m"
        )


if __name__ == "__main__":
    import random

    random.seed(42)
    demo()
