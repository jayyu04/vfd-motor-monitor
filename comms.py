from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
CommMode = Literal["MOCK", "MODBUS"]
PowerState = Literal["ON", "OFF"]
FaultType = Literal["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR", "AUTO"]


# ---------------------------------------------------------------------------
# 輸出資料結構
# ---------------------------------------------------------------------------
@dataclass
class RawSignal:
    """從 VFD 讀取的原始訊號，只有三個感測值。"""

    frequency_hz: float
    current_a: float
    voltage_v: float


# ---------------------------------------------------------------------------
# VFD 通訊層
# ---------------------------------------------------------------------------
class VFDComms:
    """
    VFD 與電腦之間的通訊介面。

    MOCK  模式：從 VFD_simulator 讀取模擬訊號（開發 / 展示用）
    MODBUS 模式：透過 Modbus RTU 從真實 VFD 讀取訊號（未來實作）

    換硬體只需改這個類別，其他模組完全不動。
    """

    def __init__(self, mode: CommMode = "MOCK") -> None:
        self._mode = mode

        if self._mode == "MOCK":
            from VFD_simulator import generate as _vfd_generate
            from VFD_simulator import get_auto_current_fault as _get_auto_fault

            self._generate = _vfd_generate
            self._get_auto_fault = _get_auto_fault

        elif self._mode == "MODBUS":
            # 未來實作：
            # import minimalmodbus
            # self._instrument = minimalmodbus.Instrument(port, slave_address)
            # self._instrument.serial.baudrate = 9600
            raise NotImplementedError(
                "MODBUS 模式尚未實作，請安裝 minimalmodbus 並設定 port 和 slave address"
            )

    # ── 讀取訊號 ──────────────────────────────────────────────────────────

    def read(
        self,
        power_state: PowerState,
        fault_type: FaultType,
        elapsed_sec: float,
    ) -> RawSignal:
        """
        從 VFD 讀取原始訊號。

        Args:
            power_state: ON / OFF（由 control.py 傳入）
            fault_type:  工況（由 control.py 傳入）
            elapsed_sec: 開機後秒數（由 control.py 傳入）

        Returns:
            RawSignal（Hz / A / V）
        """
        if self._mode == "MOCK":
            raw = self._generate(
                power_state=power_state,
                fault_type=fault_type,
                elapsed_sec=elapsed_sec,
            )
            return RawSignal(
                frequency_hz=raw.frequency_hz,
                current_a=raw.current_a,
                voltage_v=raw.voltage_v,
            )

        elif self._mode == "MODBUS":
            # 未來實作：
            # freq = self._instrument.read_register(0x2100, decimals=2)
            # curr = self._instrument.read_register(0x2101, decimals=2)
            # volt = self._instrument.read_register(0x2102, decimals=1)
            # return RawSignal(freq, curr, volt)
            raise NotImplementedError("MODBUS 模式尚未實作")

    # ── AUTO 模式輔助 ──────────────────────────────────────────────────────

    def get_auto_current_fault(self) -> str:
        """
        AUTO 模式下回傳目前實際模擬的工況。
        給 dashboard 顯示「實際工況 vs ML 判斷」用。
        """
        if self._mode == "MOCK":
            return self._get_auto_fault()
        return "UNKNOWN"

    # ── 連線狀態 ──────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """
        檢查通訊是否正常。
        MOCK 模式永遠回傳 True。
        MODBUS 模式未來實作實體連線檢查。
        """
        if self._mode == "MOCK":
            return True
        # 未來實作：
        # return self._instrument.serial.isOpen()
        return False


# ---------------------------------------------------------------------------
# 全域單例
# ---------------------------------------------------------------------------
_comms: VFDComms | None = None


def get_comms(mode: CommMode = "MOCK") -> VFDComms:
    """取得全域 VFDComms 實例（單例模式）。"""
    global _comms
    if _comms is None:
        _comms = VFDComms(mode=mode)
    return _comms
