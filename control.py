from __future__ import annotations

"""
control.py — 系統唯一控制入口

職責分兩塊：

1. 模擬控制（現在）
   - 控制 simulator 產生哪種異常資料
   - 控制電源開關、目標頻率
   - dashboard 只跟這裡說話

2. 硬體控制（未來換真實設備時）
   - 替換 _send_freq_command() 實作
   - 接入 Modbus RTU / RS-485
   - 其他模組完全不用動

資料流：
    dashboard → control.py → simulator / 真實硬體
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

from database import init_db, insert_record
from main import MotorMonitor
from ml_model import load_model
from simulator import FaultType, PowerState, STARTUP_DURATION_SEC

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
ControlMode = Literal["MOCK", "MODBUS"]

# ---------------------------------------------------------------------------
# 頻率保護範圍
# ---------------------------------------------------------------------------
MIN_FREQ_HZ = 0.0
MAX_FREQ_HZ = 60.0


# ---------------------------------------------------------------------------
# 指令記錄
# ---------------------------------------------------------------------------
@dataclass
class ControlCommand:
    timestamp: str
    command_type: str  # SET_FREQ / SET_POWER / SET_FAULT / STOP / RESET
    value: str  # 指令內容（頻率數值 or ON/OFF or fault_type）
    success: bool
    note: str = ""


# ---------------------------------------------------------------------------
# 主控制器
# ---------------------------------------------------------------------------
class SystemController:
    """
    系統唯一控制入口。

    使用方式（dashboard 呼叫）：
        ctrl = SystemController()
        ctrl.power_on()
        ctrl.set_fault_type("STALL")
        ctrl.set_frequency(45.0)

        record = ctrl.tick()          # 產生一筆資料並寫入資料庫
        status = ctrl.status()        # 取得目前系統狀態
    """

    def __init__(self, mode: ControlMode = "MOCK"):
        self._mode = mode
        self._monitor = MotorMonitor(interval_sec=1.0)
        self._target_hz = 50.0
        self._startup_begin: Optional[float] = None
        self._cmd_history: List[ControlCommand] = []

    # ── 初始化 ──

    def initialize(self) -> None:
        """系統啟動時呼叫一次，載入 ML 模型並初始化資料庫。"""
        load_model()
        init_db()

    # ── 模擬控制（第一塊職責）──

    def power_on(self, note: str = "") -> bool:
        """開機：重置啟動計時，通知 monitor。"""
        self._startup_begin = time.time()
        self._monitor.set_power("ON")
        self._monitor._state.startup_elapsed_sec = 0.0
        self._log("SET_POWER", "ON", True, note)
        return True

    def power_off(self, note: str = "") -> bool:
        """關機：重置所有狀態。"""
        self._startup_begin = None
        self._monitor.set_power("OFF")
        self._monitor._state.startup_elapsed_sec = 0.0

        # 同時送停機指令給硬體（MOCK 下直接成功）
        self._send_freq_command(0.0)
        self._log("SET_POWER", "OFF", True, note)
        return True

    def set_fault_type(self, fault_type: FaultType, note: str = "") -> bool:
        """
        切換模擬異常模式（測試用）。
        只在 RUNNING 狀態下生效，STARTUP 期間不切換。
        """
        ms = self.machine_state
        if ms == "RUNNING":
            self._monitor.set_fault_type(fault_type)
            self._log("SET_FAULT", fault_type, True, note)
            return True

        self._log("SET_FAULT", fault_type, False, f"拒絕：目前狀態={ms}")
        return False

    def set_frequency(self, hz: float, note: str = "") -> bool:
        """
        設定目標頻率。
        模擬模式：只記錄指令（simulator 自己決定頻率範圍）
        真實硬體：送出 Modbus 指令
        """
        hz = max(MIN_FREQ_HZ, min(MAX_FREQ_HZ, hz))
        self._target_hz = hz
        success = self._send_freq_command(hz)
        self._log("SET_FREQ", f"{hz:.1f}Hz", success, note)
        return success

    def reset_fault(self, note: str = "") -> bool:
        """故障重置指令。"""
        success = self._send_reset_command()
        self._log("RESET", "FAULT_RESET", success, note)
        return success

    # ── 資料產生（main.py 邏輯整合進來）──

    def tick(self) -> Optional[object]:
        """
        產生一筆資料並寫入資料庫。
        關機狀態或 monitor 未運轉時回傳 None。
        """
        if self.power_state == "OFF":
            return None

        # 同步 startup_elapsed
        self._monitor._state.startup_elapsed_sec = self.startup_elapsed_sec

        record = self._monitor.tick()
        insert_record(record)
        return record

    # ── 狀態屬性 ──

    @property
    def power_state(self) -> PowerState:
        return self._monitor._state.power_state

    @property
    def machine_state(self) -> str:
        if self.power_state == "OFF":
            return "OFF"
        return (
            "STARTUP" if self.startup_elapsed_sec < STARTUP_DURATION_SEC else "RUNNING"
        )

    @property
    def startup_elapsed_sec(self) -> float:
        if self._startup_begin is None or self.power_state == "OFF":
            return 0.0
        return max(0.0, time.time() - self._startup_begin)

    @property
    def current_fault_type(self) -> FaultType:
        return self._monitor._state.fault_type

    @property
    def target_hz(self) -> float:
        return self._target_hz

    @property
    def is_running(self) -> bool:
        return self.power_state == "ON"

    def status(self) -> dict:
        return {
            "mode": self._mode,
            "power_state": self.power_state,
            "machine_state": self.machine_state,
            "fault_type": self.current_fault_type,
            "target_hz": self._target_hz,
            "startup_elapsed": round(self.startup_elapsed_sec, 1),
            "is_running": self.is_running,
            "cmd_count": len(self._cmd_history),
        }

    @property
    def cmd_history(self) -> List[ControlCommand]:
        return self._cmd_history

    # ── 硬體通訊層（第二塊職責，未來換真實硬體只改這裡）──

    def _send_freq_command(self, hz: float) -> bool:
        """
        送出頻率指令給變頻器。

        目前 MOCK：直接回傳成功。

        未來 Modbus 實作：
            import minimalmodbus
            inst = minimalmodbus.Instrument('/dev/ttyUSB0', 1)
            inst.write_register(0x2000, int(hz * 100))
            return True
        """
        if self._mode == "MOCK":
            return True
        raise NotImplementedError("Modbus 通訊尚未實作")

    def _send_reset_command(self) -> bool:
        """
        送出故障重置指令。

        目前 MOCK：直接回傳成功。

        未來 Modbus 實作：
            inst.write_register(0x2001, 0x0010)  # 重置指令碼
            return True
        """
        if self._mode == "MOCK":
            return True
        raise NotImplementedError("Modbus 通訊尚未實作")

    # ── 內部工具 ──

    def _log(self, cmd_type: str, value: str, success: bool, note: str) -> None:
        self._cmd_history.append(
            ControlCommand(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                command_type=cmd_type,
                value=value,
                success=success,
                note=note,
            )
        )


# ---------------------------------------------------------------------------
# 全域控制器（dashboard.py / 其他模組 import 使用）
# ---------------------------------------------------------------------------
_controller: Optional[SystemController] = None


def get_controller() -> SystemController:
    """取得全域控制器，第一次呼叫時初始化。"""
    global _controller
    if _controller is None:
        _controller = SystemController(mode="MOCK")
        _controller.initialize()
    return _controller


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo() -> None:
    import random

    random.seed(None)

    print("=" * 60)
    print("SystemController Demo")
    print("=" * 60)

    ctrl = SystemController(mode="MOCK")
    ctrl.initialize()

    print("\n── 開機流程 ──")
    ctrl.power_on(note="Dashboard 按下開機")
    time.sleep(0.1)
    print(f"  power_state={ctrl.power_state}")
    print(f"  machine_state={ctrl.machine_state}")

    print("\n── 設定頻率 ──")
    ctrl.set_frequency(45.0, note="設定 45Hz")
    print(f"  target_hz={ctrl.target_hz}")

    print("\n── 模擬等待啟動完成後切換模式 ──")
    # 強制設 elapsed 跳過啟動期
    ctrl._monitor._state.startup_elapsed_sec = 10.0
    ctrl._startup_begin = time.time() - 10.0

    for ft in ["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS"]:
        ctrl.set_fault_type(ft)
        record = ctrl.tick()
        if record:
            print(
                f"  fault={ft:<14} "
                f"A={record.current_a:5.1f}  "
                f"rule={record.rule_fault_type:<14} "
                f"ml={record.ml_fault_type:<14} "
                f"final={record.final_level}"
            )

    print("\n── 關機 ──")
    ctrl.power_off(note="Dashboard 按下關機")
    print(f"  power_state={ctrl.power_state}")
    print(f"  machine_state={ctrl.machine_state}")

    print("\n── 指令歷史 ──")
    for cmd in ctrl.cmd_history:
        print(
            f"  {cmd.timestamp}  {cmd.command_type:<12} {cmd.value:<20} success={cmd.success}"
        )

    print("\n── 系統狀態 ──")
    for k, v in ctrl.status().items():
        print(f"  {k:<20} {v}")


if __name__ == "__main__":
    demo()
