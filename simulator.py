from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Generator, Literal, Optional

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
PowerState = Literal["ON", "OFF"]
MachineState = Literal["OFF", "STARTUP", "RUNNING"]
FaultType = Literal["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR"]

# ---------------------------------------------------------------------------
# 馬達固定規格（380V / 10HP / 4極 / 50Hz）
# ---------------------------------------------------------------------------
MOTOR_ID = "MOTOR-001"
VOLTAGE_V = 380.0
RATED_FREQ_HZ = 50.0
RATED_CURRENT = 15.0  # FLA (Full Load Amps)
POLES = 4

STARTUP_DURATION_SEC = 3.0

# ---------------------------------------------------------------------------
# 各 fault_type 的穩態中心值與允許浮動範圍
#
# 設計原則：
#   - center      : 穩定運轉時的中心值（第一筆資料用隨機取，之後在此附近浮動）
#   - drift_hz    : 每筆頻率最大漂移量（真實變頻器運轉中頻率非常穩定）
#   - drift_amp   : 每筆電流最大漂移量
#   - BEARING_WEAR 的 drift_amp 刻意做大，模擬電流漣波特徵
# ---------------------------------------------------------------------------
MODE_CONFIG = {
    "NORMAL": {
        "freq_center": (30.0, 60.0),  # 中心值隨機範圍
        "amp_center": (10.5, 14.0),  # 下限拉高，與 BEARING_WEAR（6~9A）明確區隔
        "drift_hz": 0.2,
        "drift_amp": 0.3,
    },
    "OVERLOAD": {
        "freq_center": (45.0, 50.0),
        "amp_center": (18.5, 22.0),  # 明確超過 110% FLA，與 BEARING_WEAR 區隔
        "drift_hz": 0.2,
        "drift_amp": 0.4,
    },
    "STALL": {
        "freq_center": (20.0, 35.0),  # 頻率往上但轉速拉不起來
        "amp_center": (24.0, 30.0),  # 拉高確保穩定觸發 CRITICAL
        "drift_hz": 0.3,
        "drift_amp": 0.5,
    },
    "LOAD_LOSS": {
        "freq_center": (45.0, 50.0),  # 確保頻率 > 40Hz
        "amp_center": (2.0, 4.0),  # 確保電流 < 5A
        "drift_hz": 0.2,
        "drift_amp": 0.2,
    },
    "BEARING_WEAR": {
        "freq_center": (45.0, 50.0),  # 頻率穩定
        "amp_center": (10.0, 13.0),  # 電流中心值正常，靠 std 跟 NORMAL 區分
        "drift_hz": 0.15,
        "drift_amp": 2.5,  # 電流波動大，模擬漣波特徵
    },
}


# ---------------------------------------------------------------------------
# 資料結構
# 輸出只有三個原始感測值，其餘由 device_adapter.py 推算
# ---------------------------------------------------------------------------
@dataclass
class RawSensorRecord:
    timestamp: str
    motor_id: str
    power_state: PowerState
    machine_state: MachineState
    fault_type: FaultType  # 模擬用輸入，非診斷結果

    # 三個原始感測值（device_adapter 接收這三個）
    frequency_hz: float
    current_a: float
    voltage_v: float


# ---------------------------------------------------------------------------
# 內部狀態：維持連續性
# 用 _State 儲存上一筆的數值，讓每筆資料在前一筆基礎上小幅浮動
# ---------------------------------------------------------------------------
@dataclass
class _State:
    frequency_hz: float
    current_a: float
    initialized: bool = False


_motor_state: Dict[str, _State] = {}


def _get_state(motor_id: str) -> _State:
    if motor_id not in _motor_state:
        _motor_state[motor_id] = _State(frequency_hz=0.0, current_a=0.0)
    return _motor_state[motor_id]


def _reset_state(motor_id: str) -> None:
    """關機或重新啟動時重置狀態"""
    _motor_state[motor_id] = _State(frequency_hz=0.0, current_a=0.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 各狀態的記錄建構
# ---------------------------------------------------------------------------


def _build_off_record(motor_id: str, fault_type: FaultType) -> RawSensorRecord:
    _reset_state(motor_id)
    return RawSensorRecord(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        motor_id=motor_id,
        power_state="OFF",
        machine_state="OFF",
        fault_type=fault_type,
        frequency_hz=0.0,
        current_a=0.0,
        voltage_v=0.0,
    )


def _build_startup_record(
    motor_id: str, fault_type: FaultType, elapsed: float
) -> RawSensorRecord:
    """
    啟動過渡期（0 ~ STARTUP_DURATION_SEC）：
    - 頻率從低往上爬，依 elapsed 線性增加
    - 電流為額定的 4~6 倍（直接啟動特性）
    - 啟動結束後電流才恢復正常，rules.py 需做啟動遮蔽
    """
    progress = elapsed / STARTUP_DURATION_SEC  # 0.0 ~ 1.0

    # 頻率從 5Hz 線性爬升到目標頻率附近
    cfg = MODE_CONFIG[fault_type]
    target_freq = sum(cfg["freq_center"]) / 2.0
    frequency_hz = round(5.0 + progress * (target_freq - 5.0), 2)

    # 啟動電流：變頻器限流，峰值 1.5~2 倍額定（22~30A）
    # 隨啟動完成線性降到穩態電流中心值附近
    cfg = MODE_CONFIG[fault_type]
    peak_current = RATED_CURRENT * random.uniform(1.5, 2.0)
    target_current = sum(cfg["amp_center"]) / 2.0
    current_a = round(peak_current - progress * (peak_current - target_current), 2)

    state = _get_state(motor_id)
    state.frequency_hz = frequency_hz
    state.current_a = current_a
    state.initialized = False  # 啟動期間不算穩態初始化

    return RawSensorRecord(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        motor_id=motor_id,
        power_state="ON",
        machine_state="STARTUP",
        fault_type=fault_type,
        frequency_hz=frequency_hz,
        current_a=current_a,
        voltage_v=VOLTAGE_V,
    )


def _build_running_record(motor_id: str, fault_type: FaultType) -> RawSensorRecord:
    """
    穩定運轉：
    - 第一筆在中心值範圍內隨機取初始值
    - 之後每筆在上一筆基礎上小幅漂移（連續性）
    - BEARING_WEAR 的電流漂移量刻意放大模擬漣波
    """
    cfg = MODE_CONFIG[fault_type]
    state = _get_state(motor_id)

    if not state.initialized:
        # 第一筆：隨機取中心值
        state.frequency_hz = round(random.uniform(*cfg["freq_center"]), 2)
        state.current_a = round(random.uniform(*cfg["amp_center"]), 2)
        state.initialized = True
    else:
        # 後續：小幅漂移
        d_hz = random.uniform(-cfg["drift_hz"], cfg["drift_hz"])
        d_amp = random.uniform(-cfg["drift_amp"], cfg["drift_amp"])

        state.frequency_hz = round(
            _clamp(
                state.frequency_hz + d_hz,
                cfg["freq_center"][0] - 1.0,
                cfg["freq_center"][1] + 1.0,
            ),
            2,
        )
        state.current_a = round(
            _clamp(
                state.current_a + d_amp,
                cfg["amp_center"][0] - 1.5,
                cfg["amp_center"][1] + 1.5,
            ),
            2,
        )

    return RawSensorRecord(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        motor_id=motor_id,
        power_state="ON",
        machine_state="RUNNING",
        fault_type=fault_type,
        frequency_hz=state.frequency_hz,
        current_a=state.current_a,
        voltage_v=VOLTAGE_V,
    )


# ---------------------------------------------------------------------------
# 對外主入口
# ---------------------------------------------------------------------------


def generate_sensor_record(
    power_state: PowerState,
    fault_type: FaultType,
    startup_elapsed_sec: float,
    motor_id: str = MOTOR_ID,
) -> RawSensorRecord:
    """
    規則：
    - OFF → 全零，重置狀態
    - ON + elapsed < STARTUP_DURATION_SEC → 啟動過渡資料
    - ON + elapsed >= STARTUP_DURATION_SEC → 依 fault_type 穩態運轉資料
    """
    if power_state == "OFF":
        return _build_off_record(motor_id=motor_id, fault_type=fault_type)

    if startup_elapsed_sec < STARTUP_DURATION_SEC:
        return _build_startup_record(
            motor_id=motor_id, fault_type=fault_type, elapsed=startup_elapsed_sec
        )

    return _build_running_record(motor_id=motor_id, fault_type=fault_type)


def generate_sensor_stream(
    power_state: PowerState,
    fault_type: FaultType,
    startup_elapsed_sec: float,
    motor_id: str = MOTOR_ID,
    interval_sec: float = 1.0,
    max_records: Optional[int] = None,
) -> Generator[RawSensorRecord, None, None]:
    """準即時資料流，每 interval_sec 秒產生一筆。"""
    count = 0
    elapsed = startup_elapsed_sec

    while True:
        yield generate_sensor_record(
            power_state=power_state,
            fault_type=fault_type,
            startup_elapsed_sec=elapsed,
            motor_id=motor_id,
        )

        count += 1
        if power_state == "ON":
            elapsed += interval_sec

        if max_records is not None and count >= max_records:
            break

        time.sleep(interval_sec)


def record_to_dict(record: RawSensorRecord) -> Dict[str, object]:
    return asdict(record)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo() -> None:
    test_cases: list[tuple[PowerState, FaultType, float]] = [
        ("OFF", "NORMAL", 0.0),
        ("ON", "NORMAL", 0.0),  # 啟動初期
        ("ON", "NORMAL", 1.5),  # 啟動中段
        ("ON", "NORMAL", 4.0),  # 正常運轉（連續5筆看連續性）
        ("ON", "OVERLOAD", 4.0),
        ("ON", "STALL", 4.0),
        ("ON", "LOAD_LOSS", 4.0),
        ("ON", "BEARING_WEAR", 4.0),
    ]

    for power_state, fault_type, startup_elapsed_sec in test_cases:
        print("=" * 70)
        print(
            f"power_state={power_state}  fault_type={fault_type}  "
            f"elapsed={startup_elapsed_sec}s"
        )

        if fault_type == "NORMAL" and startup_elapsed_sec == 4.0:
            # 連續5筆看漂移效果
            print("  >>> 連續 5 筆 <<<")
            for _ in range(5):
                r = generate_sensor_record(power_state, fault_type, startup_elapsed_sec)
                print(
                    f"  Hz={r.frequency_hz:6.2f}  "
                    f"A={r.current_a:6.2f}  "
                    f"V={r.voltage_v}"
                )
        else:
            r = generate_sensor_record(power_state, fault_type, startup_elapsed_sec)
            print(
                f"  Hz={r.frequency_hz:6.2f}  "
                f"A={r.current_a:6.2f}  "
                f"V={r.voltage_v}"
            )


if __name__ == "__main__":
    random.seed(42)
    demo()
