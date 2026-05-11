from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

from config import STARTUP_DURATION_SEC, MIN_FREQ_HZ, MAX_FREQ_HZ
from main import get_monitor

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
PowerState = Literal["ON", "OFF"]
ControlMode = Literal["MOCK", "MODBUS"]


# ---------------------------------------------------------------------------
# 指令記錄
# ---------------------------------------------------------------------------
@dataclass
class ControlCommand:
    timestamp: str
    command_type: str
    value: str
    success: bool
    note: str = ""


# ---------------------------------------------------------------------------
# 主控制器
# ---------------------------------------------------------------------------
class SystemController:
    """
    系統唯一控制入口。

    職責：
        - 管理系統狀態（power_state / elapsed_sec / fault_type）
        - 接收 dashboard 的指令（開機/關機/選工況）
        - 通知 main.py 執行資料流水線
        - 未來換真實硬體只改 _send_freq_command()

    dashboard 只跟這裡說話。
    """

    def __init__(self, mode: ControlMode = "MOCK") -> None:
        self._mode = mode
        self._power_state: PowerState = "OFF"
        self._fault_type: str = "NORMAL"
        self._startup_begin: Optional[float] = None
        self._target_hz: float = 50.0
        self._cmd_history: List[ControlCommand] = []
        self._monitor = get_monitor()

    # ── 開關機 ──────────────────────────────────────────────────────────────

    def power_on(self, note: str = "") -> bool:
        """開機：記錄開機時間，設定狀態為 ON。"""
        self._power_state = "ON"
        self._startup_begin = time.time()
        self._log("SET_POWER", "ON", True, note)
        return True

    def power_off(self, note: str = "") -> bool:
        """關機：重置所有狀態，清除 main 的窗口。"""
        self._power_state = "OFF"
        self._startup_begin = None
        self._send_freq_command(0.0)
        self._monitor.on_power_off()
        self._log("SET_POWER", "OFF", True, note)
        return True

    # ── 工況設定 ──────────────────────────────────────────────────────────

    def set_fault_type(self, fault_type: str, note: str = "") -> bool:
        """
        切換模擬工況。
        開機狀態下隨時可切換，包含 AUTO 模式。
        """
        self._fault_type = fault_type
        self._log("SET_FAULT", fault_type, True, note)
        return True

    def set_frequency(self, hz: float, note: str = "") -> bool:
        """
        設定目標頻率。
        模擬模式：只記錄指令。
        真實硬體：送出 Modbus 指令。
        """
        hz = max(MIN_FREQ_HZ, min(MAX_FREQ_HZ, hz))
        self._target_hz = hz
        success = self._send_freq_command(hz)
        self._log("SET_FREQ", f"{hz:.1f}Hz", success, note)
        return success

    # ── 資料流觸發 ──────────────────────────────────────────────────────────

    def tick(self) -> Optional[object]:
        """
        dashboard 定時呼叫這裡。
        control 把目前狀態帶給 main，main 執行一次完整流水線。
        關機狀態回傳 None。
        """
        if self._power_state == "OFF":
            return None

        return self._monitor.tick(
            power_state=self._power_state,
            fault_type=self._fault_type,
            elapsed_sec=self.elapsed_sec,
        )

    # ── 狀態屬性 ──────────────────────────────────────────────────────────

    @property
    def power_state(self) -> PowerState:
        return self._power_state

    @property
    def elapsed_sec(self) -> float:
        """開機後經過的秒數。關機時回傳 0。"""
        if self._startup_begin is None or self._power_state == "OFF":
            return 0.0
        return max(0.0, time.time() - self._startup_begin)

    @property
    def machine_state(self) -> str:
        """
        設備階段，供 dashboard 顯示用。
        OFF → 關機
        啟動中 → elapsed < STARTUP_DURATION_SEC
        運轉中 → elapsed >= STARTUP_DURATION_SEC
        """
        if self._power_state == "OFF":
            return "關機"
        if self.elapsed_sec < STARTUP_DURATION_SEC:
            return "啟動中"
        return "運轉中"

    @property
    def fault_type(self) -> str:
        return self._fault_type

    @property
    def target_hz(self) -> float:
        return self._target_hz

    @property
    def is_running(self) -> bool:
        return self._power_state == "ON"

    @property
    def startup_elapsed_sec(self) -> float:
        """向下相容的別名。"""
        return self.elapsed_sec

    def status(self) -> dict:
        return {
            "mode": self._mode,
            "power_state": self._power_state,
            "machine_state": self.machine_state,
            "fault_type": self._fault_type,
            "target_hz": self._target_hz,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "is_running": self.is_running,
        }

    @property
    def cmd_history(self) -> List[ControlCommand]:
        return self._cmd_history

    # ── 硬體通訊層（未來換真實硬體只改這裡）──────────────────────────────

    def _send_freq_command(self, hz: float) -> bool:
        """
        送出頻率指令給變頻器。

        MOCK 模式：直接回傳成功。
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

        MOCK 模式：直接回傳成功。
        未來 Modbus 實作：
            inst.write_register(0x2001, 0x0010)
            return True
        """
        if self._mode == "MOCK":
            return True
        raise NotImplementedError("Modbus 通訊尚未實作")

    # ── 內部工具 ──────────────────────────────────────────────────────────

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
# 全域控制器
# ---------------------------------------------------------------------------
_controller: Optional[SystemController] = None


def get_controller() -> SystemController:
    """取得全域控制器，第一次呼叫時初始化。"""
    global _controller
    if _controller is None:
        _controller = SystemController(mode="MOCK")
    return _controller
