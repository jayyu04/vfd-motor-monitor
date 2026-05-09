from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Literal, Optional

from device_adapter import AdaptedRecord, adapt
from ml_model import MlResult, load_model, predict
from rules import RuleResult, evaluate
from simulator import FaultType, PowerState, generate_sensor_record

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
ControlMode = Literal["SIMULATE", "REAL"]  # 模擬模式 or 真實硬體


# ---------------------------------------------------------------------------
# 完整紀錄（最終輸出給 database 和 dashboard）
# ---------------------------------------------------------------------------
@dataclass
class FullRecord:
    # ── 來自 AdaptedRecord ──
    timestamp: str
    motor_id: str
    power_state: str
    machine_state: str
    fault_type: str  # simulator 輸入的模擬模式（測試用）

    frequency_hz: float
    current_a: float
    voltage_v: float
    sync_rpm: float
    slip_ratio: float
    rpm_est: float
    power_factor: float
    input_power_kw: float
    output_power_kw: float
    torque_nm: float

    # ── 來自 rules.py ──
    rule_fault_type: str
    rule_level: str
    rule_score: int
    rule_reasons: List[str]
    rule_is_startup: bool

    # ── 來自 ml_model.py ──
    ml_fault_type: str
    ml_level: str
    ml_confidence: float
    ml_probabilities: Dict[str, float]

    # ── 綜合風險等級（rules 和 ml 取較嚴重的）──
    final_level: str


# ---------------------------------------------------------------------------
# 風險等級比較工具
# ---------------------------------------------------------------------------
_LEVEL_ORDER = {"NORMAL": 0, "WARNING": 1, "DANGER": 2, "CRITICAL": 3}


def _higher_level(a: str, b: str) -> str:
    """回傳兩個等級中較嚴重的那個"""
    return a if _LEVEL_ORDER.get(a, 0) >= _LEVEL_ORDER.get(b, 0) else b


# ---------------------------------------------------------------------------
# 系統狀態
# ---------------------------------------------------------------------------
@dataclass
class SystemState:
    power_state: PowerState = "OFF"
    fault_type: FaultType = "NORMAL"
    startup_elapsed_sec: float = 0.0
    motor_id: str = "MOTOR-001"
    running: bool = False


# ---------------------------------------------------------------------------
# 主控制器
# ---------------------------------------------------------------------------
class MotorMonitor:
    """
    主流程控制器。

    使用方式：
        monitor = MotorMonitor()
        monitor.start()

        # dashboard 控制
        monitor.set_power("ON")
        monitor.set_fault_type("STALL")

        # 取得最新紀錄
        record = monitor.latest_record

        monitor.stop()
    """

    def __init__(
        self,
        interval_sec: float = 1.0,
        on_record: Optional[Callable[[FullRecord], None]] = None,
    ):
        """
        interval_sec : 每筆資料的間隔秒數
        on_record    : 每產生一筆 FullRecord 時的 callback（給 dashboard 用）
        """
        self._interval = interval_sec
        self._on_record = on_record
        self._state = SystemState()
        self._latest: Optional[FullRecord] = None
        self._history: List[FullRecord] = []
        self._running = False

    # ── 控制介面（dashboard 呼叫）──

    def set_power(self, state: PowerState) -> None:
        if state == "ON" and self._state.power_state == "OFF":
            self._state.startup_elapsed_sec = 0.0  # 重置啟動計時
        if state == "OFF":
            self._state.startup_elapsed_sec = 0.0
        self._state.power_state = state

    def set_fault_type(self, fault_type: FaultType) -> None:
        """切換模擬的異常模式（測試用）"""
        self._state.fault_type = fault_type

    # ── 屬性 ──

    @property
    def latest_record(self) -> Optional[FullRecord]:
        return self._latest

    @property
    def history(self) -> List[FullRecord]:
        return self._history

    # ── 核心：單筆資料處理 ──

    def _process_one(self) -> FullRecord:
        """
        執行一個完整的資料處理週期：
        simulator → device_adapter → rules + ml → FullRecord
        """
        state = self._state

        # 1. simulator 產生原始資料
        raw = generate_sensor_record(
            power_state=state.power_state,
            fault_type=state.fault_type,
            startup_elapsed_sec=state.startup_elapsed_sec,
            motor_id=state.motor_id,
        )

        # 2. device_adapter 推算完整物理量
        adapted: AdaptedRecord = adapt(raw)

        # 3. rules 判斷
        rule_result: RuleResult = evaluate(adapted)

        # 4. ml 推論
        ml_result: MlResult = predict(adapted)

        # 5. 綜合風險等級（取較嚴重的）
        final_level = _higher_level(rule_result.level, ml_result.level)

        # 6. 組合 FullRecord
        record = FullRecord(
            # AdaptedRecord 欄位
            timestamp=adapted.timestamp,
            motor_id=adapted.motor_id,
            power_state=adapted.power_state,
            machine_state=adapted.machine_state,
            fault_type=adapted.fault_type,
            frequency_hz=adapted.frequency_hz,
            current_a=adapted.current_a,
            voltage_v=adapted.voltage_v,
            sync_rpm=adapted.sync_rpm,
            slip_ratio=adapted.slip_ratio,
            rpm_est=adapted.rpm_est,
            power_factor=adapted.power_factor,
            input_power_kw=adapted.input_power_kw,
            output_power_kw=adapted.output_power_kw,
            torque_nm=adapted.torque_nm,
            # rules 結果
            rule_fault_type=rule_result.fault_type,
            rule_level=rule_result.level,
            rule_score=rule_result.anomaly_score,
            rule_reasons=rule_result.reasons,
            rule_is_startup=rule_result.is_startup,
            # ml 結果
            ml_fault_type=ml_result.fault_type,
            ml_level=ml_result.level,
            ml_confidence=ml_result.confidence,
            ml_probabilities=ml_result.probabilities,
            # 綜合
            final_level=final_level,
        )

        # 7. 更新啟動計時
        if state.power_state == "ON":
            state.startup_elapsed_sec += self._interval

        return record

    # ── 執行模式 ──

    def tick(self) -> FullRecord:
        """
        單步執行一筆（dashboard 用 polling 模式時呼叫）。
        """
        record = self._process_one()
        self._latest = record
        self._history.append(record)
        if self._on_record:
            self._on_record(record)
        return record

    def run_loop(self, max_ticks: Optional[int] = None) -> None:
        """
        持續執行迴圈（背景執行緒用）。
        dashboard 啟動後在背景 thread 呼叫這個。
        """
        self._running = True
        count = 0
        while self._running:
            self.tick()
            count += 1
            if max_ticks is not None and count >= max_ticks:
                break
            time.sleep(self._interval)

    def stop(self) -> None:
        self._running = False


def record_to_dict(record: FullRecord) -> Dict:
    return asdict(record)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo() -> None:
    import random

    random.seed(None)

    print("載入 ML 模型...")
    load_model()

    print("\n" + "=" * 60)
    print("Demo：模擬各種異常模式，每種跑 3 筆")
    print("=" * 60)

    monitor = MotorMonitor(interval_sec=1.0)

    fault_types: List[FaultType] = [
        "NORMAL",
        "OVERLOAD",
        "STALL",
        "LOAD_LOSS",
        "BEARING_WEAR",
    ]

    for fault_type in fault_types:
        print(f"\n── {fault_type} ──")
        monitor.set_power("ON")
        monitor.set_fault_type(fault_type)
        monitor._state.startup_elapsed_sec = 10.0  # 跳過啟動期

        for _ in range(3):
            rec = monitor.tick()
            print(
                f"  Hz={rec.frequency_hz:5.1f}  "
                f"A={rec.current_a:5.1f}  "
                f"rule={rec.rule_fault_type:<14} score={rec.rule_score:>3}  "
                f"ml={rec.ml_fault_type:<14} conf={rec.ml_confidence:.0%}  "
                f"→ final={rec.final_level}"
            )

    print("\n── 啟動過渡期測試 ──")
    monitor.set_power("OFF")
    monitor.tick()  # OFF 狀態
    monitor.set_power("ON")
    monitor._state.startup_elapsed_sec = 0.0

    for i in range(4):
        rec = monitor.tick()
        print(
            f"  elapsed={monitor._state.startup_elapsed_sec - 1.0:.1f}s  "
            f"state={rec.machine_state:<8}  "
            f"A={rec.current_a:5.1f}  "
            f"rule={rec.rule_fault_type:<8}  "
            f"→ final={rec.final_level}"
        )


if __name__ == "__main__":
    demo()
