from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from config import (
    BEARING_FREQ_STABLE_RANGE,
    BEARING_STD_THRESHOLD,
    BEARING_WINDOW_SIZE,
    LOAD_LOSS_CURRENT_MAX,
    OVERLOAD_CURRENT_BASE,
    OVERLOAD_SCORE_PER_AMP,
    STALL_CURRENT_BASE,
    STALL_CURRENT_SCORE_PER_AMP,
    STALL_FREQ_THRESHOLD,
)
from physics import PhysicsRecord

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
RuleLevel     = Literal["NORMAL", "WARNING", "DANGER", "CRITICAL"]
RuleFaultType = Literal["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR"]

# ---------------------------------------------------------------------------
# 模式直接綁定等級（不再用分數決定等級）
# ---------------------------------------------------------------------------
FAULT_TO_LEVEL: Dict[str, RuleLevel] = {
    "NORMAL":       "NORMAL",
    "OVERLOAD":     "WARNING",
    "BEARING_WEAR": "WARNING",
    "LOAD_LOSS":    "DANGER",
    "STALL":        "CRITICAL",
}

# ---------------------------------------------------------------------------
# 判斷結果
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    fault_type:       RuleFaultType
    level:            RuleLevel
    rule_confidence:  int            # 0~100，觸發條件的信心程度
    contributions:    Dict[str, int]
    reasons:          List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 滑動窗口（BEARING_WEAR 用）
# ---------------------------------------------------------------------------
class _SlidingWindow:
    def __init__(self, size: int):
        self._size    = size
        self._current: List[float] = []
        self._freq:    List[float] = []

    def push(self, current_a: float, frequency_hz: float) -> None:
        self._current.append(current_a)
        self._freq.append(frequency_hz)
        if len(self._current) > self._size:
            self._current.pop(0)
            self._freq.pop(0)

    def ready(self) -> bool:
        return len(self._current) >= self._size

    def current_std(self) -> float:
        if not self.ready():
            return 0.0
        mean     = sum(self._current) / len(self._current)
        variance = sum((x - mean) ** 2 for x in self._current) / len(self._current)
        return variance ** 0.5

    def freq_range(self) -> float:
        if not self.ready():
            return 0.0
        return max(self._freq) - min(self._freq)

    def clear(self) -> None:
        self._current.clear()
        self._freq.clear()


_windows: Dict[str, _SlidingWindow] = {}


def _get_window(motor_id: str) -> _SlidingWindow:
    if motor_id not in _windows:
        _windows[motor_id] = _SlidingWindow(BEARING_WINDOW_SIZE)
    return _windows[motor_id]


def clear_windows() -> None:
    """關機時清除所有滑動窗口。"""
    for w in _windows.values():
        w.clear()


# ---------------------------------------------------------------------------
# 各模式信心分數計算（各自 0~100，互相獨立）
# ---------------------------------------------------------------------------

def _conf_stall(current_a: float, frequency_hz: float) -> Tuple[int, List[str]]:
    """
    STALL 信心分數：電流超過 133% FLA 且頻率低於 40Hz。
    頻率條件區分 STALL（低頻高電流）vs OVERLOAD（正常頻率高電流）。
    電流越高信心越高，滿分 100。
    """
    if current_a <= STALL_CURRENT_BASE or frequency_hz >= STALL_FREQ_THRESHOLD:
        return 0, []

    over       = current_a - STALL_CURRENT_BASE
    confidence = min(int(over * STALL_CURRENT_SCORE_PER_AMP), 100)
    reasons    = [
        f"電流 {current_a:.1f}A > {STALL_CURRENT_BASE:.1f}A (133% FLA) [{confidence}%]",
        f"頻率 {frequency_hz:.1f}Hz < {STALL_FREQ_THRESHOLD:.0f}Hz（低頻確認堵轉）",
    ]
    return confidence, reasons


def _conf_load_loss(current_a: float) -> Tuple[int, List[str]]:
    """
    LOAD_LOSS 信心分數：電流低於 5A。
    電流越低信心越高，滿分 100。
    """
    if current_a >= LOAD_LOSS_CURRENT_MAX:
        return 0, []

    deficit    = LOAD_LOSS_CURRENT_MAX - current_a
    confidence = min(int(deficit / LOAD_LOSS_CURRENT_MAX * 100), 100)
    reasons    = [f"電流 {current_a:.1f}A < {LOAD_LOSS_CURRENT_MAX:.0f}A（負載流失）[{confidence}%]"]
    return confidence, reasons


def _conf_overload(current_a: float) -> Tuple[int, List[str]]:
    """
    OVERLOAD 信心分數：電流超過 110% FLA。
    電流越高信心越高，滿分 100。
    """
    if current_a <= OVERLOAD_CURRENT_BASE:
        return 0, []

    over       = current_a - OVERLOAD_CURRENT_BASE
    confidence = min(int(over * OVERLOAD_SCORE_PER_AMP), 100)
    reasons    = [f"電流 {current_a:.1f}A > {OVERLOAD_CURRENT_BASE:.1f}A (110% FLA) [{confidence}%]"]
    return confidence, reasons


def _conf_bearing(motor_id: str, current_a: float, frequency_hz: float) -> Tuple[int, List[str]]:
    """
    BEARING_WEAR 信心分數：電流標準差超過門檻且頻率穩定。
    標準差越大信心越高，滿分 100。
    """
    window = _get_window(motor_id)
    window.push(current_a, frequency_hz)

    if not window.ready():
        return 0, []

    std        = window.current_std()
    freq_range = window.freq_range()

    if std <= BEARING_STD_THRESHOLD or freq_range >= BEARING_FREQ_STABLE_RANGE:
        return 0, []

    over       = std - BEARING_STD_THRESHOLD
    confidence = min(int(over / BEARING_STD_THRESHOLD * 100), 100)
    reasons    = [
        f"電流波動標準差 {std:.2f}A > {BEARING_STD_THRESHOLD}A [{confidence}%]",
        f"頻率穩定（變化 {freq_range:.2f}Hz）[輔助確認]",
    ]
    return confidence, reasons


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def evaluate(record: PhysicsRecord) -> RuleResult:
    """
    接收 physics.py 的完整物理量，計算各模式的 Rule Confidence 並回傳 RuleResult。

    設計原則：
    - 各模式各自計算 0~100 的信心分數，互相獨立
    - 取信心分數最高的模式作為判斷結果
    - 等級直接跟模式綁定，不再由分數決定
    - 啟動遮蔽由 main.py 負責，rules.py 不判斷狀態
    """
    stall_conf,    stall_reasons    = _conf_stall(record.current_a, record.frequency_hz)
    load_conf,     load_reasons     = _conf_load_loss(record.current_a)
    overload_conf, overload_reasons = _conf_overload(record.current_a)
    bearing_conf,  bearing_reasons  = _conf_bearing(
        record.motor_id, record.current_a, record.frequency_hz
    )

    # STALL 觸發時（低頻 + 高電流），OVERLOAD 不參與競爭
    # 因為 STALL 的頻率條件已確認是堵轉，不應讓 OVERLOAD 蓋過
    if stall_conf > 0:
        overload_conf = 0

    all_conf = {
        "stall":        stall_conf,
        "load_loss":    load_conf,
        "overload":     overload_conf,
        "bearing_wear": bearing_conf,
    }

    # 同分時的優先順序：等級較嚴重的優先
    # STALL > LOAD_LOSS > OVERLOAD > BEARING_WEAR
    PRIORITY = {"stall": 4, "load_loss": 3, "overload": 2, "bearing_wear": 1}

    if any(v > 0 for v in all_conf.values()):
        dominant   = max(all_conf, key=lambda k: (all_conf[k], PRIORITY[k]))
        confidence = all_conf[dominant]
        contributions = {dominant: confidence}

        reasons_map = {
            "stall":        stall_reasons,
            "load_loss":    load_reasons,
            "overload":     overload_reasons,
            "bearing_wear": bearing_reasons,
        }
        all_reasons = reasons_map[dominant]

        fault_map: Dict[str, RuleFaultType] = {
            "stall":        "STALL",
            "load_loss":    "LOAD_LOSS",
            "overload":     "OVERLOAD",
            "bearing_wear": "BEARING_WEAR",
        }
        fault_type = fault_map[dominant]
    else:
        fault_type    = "NORMAL"
        confidence    = 0
        contributions = {}
        all_reasons   = ["各項數值正常"]

    level = FAULT_TO_LEVEL[fault_type]

    return RuleResult(
        fault_type      = fault_type,
        level           = level,
        rule_confidence = confidence,
        contributions   = contributions,
        reasons         = all_reasons,
    )
