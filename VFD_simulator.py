from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from config import (
    MODE_CONFIG,
    MOTOR_ID,
    POLES,
    RATED_CURRENT_A,
    RATED_FREQ_HZ,
    STARTUP_CURRENT_MULTIPLIER,
    STARTUP_DURATION_SEC,
    STARTUP_FREQ_START,
    STARTUP_TORQUE_MAX_RATIO,
    VOLTAGE_V,
)

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
FaultType = Literal["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR", "AUTO"]
PowerState = Literal["ON", "OFF"]

# ---------------------------------------------------------------------------
# AUTO 模式內部狀態
# ---------------------------------------------------------------------------
_AUTO_FAULT_TYPES = ["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR"]


@dataclass
class _AutoState:
    current_fault: str = "NORMAL"
    prev_fault: str = ""
    elapsed_in_mode: float = 0.0
    duration: float = 0.0


_auto_state = _AutoState()


def _next_auto_fault() -> Tuple[str, float]:
    """隨機選下一個工況，不重複上一個。"""
    candidates = [f for f in _AUTO_FAULT_TYPES if f != _auto_state.current_fault]
    fault = random.choice(candidates)
    if fault == "BEARING_WEAR":
        duration = random.uniform(20.0, 30.0)
    else:
        duration = random.uniform(10.0, 20.0)
    return fault, duration


def _update_auto_state(dt: float = 0.3) -> str:
    """
    更新 AUTO 模式的內部狀態。
    dt：距上次呼叫的秒數（預設一個 tick 約 300ms）
    回傳目前應該模擬的 fault_type。
    """
    global _auto_state

    # 第一次進入，初始化
    if _auto_state.duration == 0.0:
        _auto_state.current_fault, _auto_state.duration = _next_auto_fault()
        _auto_state.elapsed_in_mode = 0.0

    _auto_state.elapsed_in_mode += dt

    # 超過持續時間，切換工況
    if _auto_state.elapsed_in_mode >= _auto_state.duration:
        _auto_state.prev_fault = _auto_state.current_fault
        _auto_state.current_fault, _auto_state.duration = _next_auto_fault()
        _auto_state.elapsed_in_mode = 0.0
        # 切換工況時重置連續性狀態
        _reset_running_state(_auto_state.current_fault)

    return _auto_state.current_fault


# ---------------------------------------------------------------------------
# 輸出資料結構
# ---------------------------------------------------------------------------
@dataclass
class RawSignal:
    """VFD 輸出的原始訊號，只有三個感測值。"""

    frequency_hz: float
    current_a: float
    voltage_v: float


# ---------------------------------------------------------------------------
# 內部連續性狀態
# ---------------------------------------------------------------------------
@dataclass
class _RunningState:
    frequency_hz: float = 0.0
    current_a: float = 0.0
    initialized: bool = False


_running_states: Dict[str, _RunningState] = {}


def _get_running_state(fault_type: str) -> _RunningState:
    if fault_type not in _running_states:
        _running_states[fault_type] = _RunningState()
    return _running_states[fault_type]


def _reset_running_state(fault_type: str) -> None:
    _running_states[fault_type] = _RunningState()


def _reset_all_states() -> None:
    """關機時重置所有狀態。"""
    _running_states.clear()
    global _auto_state
    _auto_state = _AutoState()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 各狀態的訊號產生
# ---------------------------------------------------------------------------


def _generate_off() -> RawSignal:
    """關機：全部輸出 0。"""
    _reset_all_states()
    return RawSignal(frequency_hz=0.0, current_a=0.0, voltage_v=0.0)


def _generate_startup(elapsed: float) -> RawSignal:
    """
    啟動過渡期：
    - 頻率從 STARTUP_FREQ_START 線性爬升到額定頻率
    - 電流為額定的 1.5~2 倍，隨啟動完成線性下降
    """
    progress = min(elapsed / STARTUP_DURATION_SEC, 1.0)
    freq = STARTUP_FREQ_START + progress * (RATED_FREQ_HZ - STARTUP_FREQ_START)
    peak = RATED_CURRENT_A * random.uniform(*STARTUP_CURRENT_MULTIPLIER)
    current = peak * (1.0 - progress * 0.6)

    return RawSignal(
        frequency_hz=round(freq, 2),
        current_a=round(current, 2),
        voltage_v=VOLTAGE_V,
    )


def _generate_running(fault_type: str) -> RawSignal:
    """
    穩態運轉：
    - 第一筆從工況中心值範圍隨機取初始值
    - 之後在上一筆基礎上小幅漂移（連續性）
    - BEARING_WEAR 漂移量刻意放大模擬電流漣波
    """
    cfg = MODE_CONFIG[fault_type]
    state = _get_running_state(fault_type)

    if not state.initialized:
        state.frequency_hz = round(random.uniform(*cfg["freq_center"]), 2)
        state.current_a = round(random.uniform(*cfg["amp_center"]), 2)
        state.initialized = True
    else:
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

    return RawSignal(
        frequency_hz=state.frequency_hz,
        current_a=state.current_a,
        voltage_v=VOLTAGE_V,
    )


# ---------------------------------------------------------------------------
# 對外主入口（control.py 呼叫這裡）
# ---------------------------------------------------------------------------


def generate(
    power_state: PowerState,
    fault_type: FaultType,
    elapsed_sec: float,
) -> RawSignal:
    """
    control.py 傳入：
        power_state  → ON / OFF
        fault_type   → 工況或 AUTO
        elapsed_sec  → 開機後秒數

    回傳：
        RawSignal（Hz / A / V）
    """
    if power_state == "OFF":
        return _generate_off()

    if elapsed_sec < STARTUP_DURATION_SEC:
        return _generate_startup(elapsed_sec)

    # RUNNING 狀態
    if fault_type == "AUTO":
        active_fault = _update_auto_state()
    else:
        active_fault = fault_type

    return _generate_running(active_fault)


def get_auto_current_fault() -> str:
    """
    AUTO 模式下，回傳目前實際模擬的工況。
    給 dashboard 顯示「目前實際工況」用。
    """
    return _auto_state.current_fault


# ---------------------------------------------------------------------------
# 目前工況查詢（dashboard 直接讀取）
# ---------------------------------------------------------------------------


def get_current_fault(fault_type: str) -> str:
    """回傳目前實際模擬的工況。
    AUTO 模式回傳系統自動切換的工況，手動模式回傳 fault_type。
    """
    if fault_type == "AUTO":
        return _auto_state.current_fault
    return fault_type
