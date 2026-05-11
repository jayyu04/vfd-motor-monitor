from __future__ import annotations

from typing import Dict, List, Literal, Optional

from config import STARTUP_DURATION_SEC
from comms import get_comms, RawSignal
from data_collector import collect
from physics import calculate
from rules import evaluate, clear_windows, RuleResult
from ml_model import predict, load_model, clear_ml_windows, MlResult
from records import FullRecord
import database

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
Level = Literal["NORMAL", "WARNING", "DANGER", "CRITICAL", "啟動中"]

LEVEL_ORDER = ["NORMAL", "WARNING", "DANGER", "CRITICAL"]


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------
def _higher_level(level_a: str, level_b: str) -> str:
    """取兩個等級中較嚴重的。"""

    def rank(lv: str) -> int:
        try:
            return LEVEL_ORDER.index(lv)
        except ValueError:
            return -1  # 啟動中等非標準等級排最低

    return level_a if rank(level_a) >= rank(level_b) else level_b


def _make_startup_record(
    timestamp: str,
    motor_id: str,
    frequency_hz: float,
    current_a: float,
    voltage_v: float,
    slip_ratio: float,
    torque_nm: float,
    sync_rpm: float,
) -> FullRecord:
    """產生啟動中的資料庫紀錄，rules / ml 欄位填啟動中 / 0。"""
    return FullRecord(
        timestamp=timestamp,
        motor_id=motor_id,
        machine_state="啟動中",
        frequency_hz=frequency_hz,
        current_a=current_a,
        voltage_v=voltage_v,
        sync_rpm=round(120.0 * frequency_hz / 4, 1),
        slip_ratio=slip_ratio,
        torque_nm=torque_nm,
        rule_fault_type="啟動中",
        rule_level="啟動中",
        rule_score=0,
        rule_reasons="啟動過渡期，暫停異常判斷",
        ml_fault_type="啟動中",
        ml_level="啟動中",
        ml_confidence=0.0,
        final_level="啟動中",
    )


# ---------------------------------------------------------------------------
# 主流水線
# ---------------------------------------------------------------------------


class MotorMonitor:
    """
    流水線串接層。
    被 control.py 通知開始/停止，定時執行一次完整的資料流。
    """

    def __init__(self) -> None:
        self._comms = get_comms("MOCK")
        load_model()
        database.init_db()

    def tick(
        self,
        power_state: str,
        fault_type: str,
        elapsed_sec: float,
    ) -> Optional[FullRecord]:
        """
        執行一次完整的資料流水線。

        Args:
            power_state: ON / OFF（來自 control.py）
            fault_type:  工況（來自 control.py）
            elapsed_sec: 開機後秒數（來自 control.py）

        Returns:
            FullRecord 或 None（關機時不產生資料）
        """
        if power_state == "OFF":
            return None

        # ── 資料流 ──
        raw = self._comms.read(power_state, fault_type, elapsed_sec)
        rec = collect(raw)
        phy = calculate(rec)

        # ── 啟動遮蔽 ──
        if elapsed_sec < STARTUP_DURATION_SEC:
            full = _make_startup_record(
                timestamp=phy.timestamp,
                motor_id=phy.motor_id,
                frequency_hz=phy.frequency_hz,
                current_a=phy.current_a,
                voltage_v=phy.voltage_v,
                slip_ratio=phy.slip_ratio,
                torque_nm=phy.torque_nm,
                sync_rpm=phy.sync_rpm,
            )
            database.insert_record(full)
            return full

        # ── 正常運轉，呼叫 rules 和 ml ──
        rule_res: RuleResult = evaluate(phy)
        ml_res: MlResult = predict(phy)

        final = _higher_level(rule_res.level, ml_res.level)

        full = FullRecord(
            timestamp=phy.timestamp,
            motor_id=phy.motor_id,
            machine_state="運轉中",
            frequency_hz=phy.frequency_hz,
            current_a=phy.current_a,
            voltage_v=phy.voltage_v,
            sync_rpm=phy.sync_rpm,
            slip_ratio=phy.slip_ratio,
            torque_nm=phy.torque_nm,
            rule_fault_type=rule_res.fault_type,
            rule_level=rule_res.level,
            rule_score=rule_res.anomaly_score,
            rule_reasons=" | ".join(rule_res.reasons),
            ml_fault_type=ml_res.fault_type,
            ml_level=ml_res.level,
            ml_confidence=ml_res.confidence,
            final_level=final,
        )

        database.insert_record(full)
        return full

    def on_power_off(self) -> None:
        """關機時清除所有狀態。"""
        clear_windows()
        clear_ml_windows()


# ---------------------------------------------------------------------------
# 全域單例
# ---------------------------------------------------------------------------
_monitor: Optional[MotorMonitor] = None


def get_monitor() -> MotorMonitor:
    """取得全域 MotorMonitor 實例（單例模式）。"""
    global _monitor
    if _monitor is None:
        _monitor = MotorMonitor()
    return _monitor
