from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from config import (
    BEARING_FREQ_STABLE_RANGE,
    BEARING_MAX_SCORE,
    BEARING_STD_THRESHOLD,
    BEARING_WINDOW_SIZE,
    LOAD_LOSS_CURRENT_MAX,
    LOAD_LOSS_FREQ_MIN,
    LOAD_LOSS_MAX_SCORE,
    OVERLOAD_CURRENT_BASE,
    OVERLOAD_MAX_SCORE,
    OVERLOAD_SCORE_PER_AMP,
    RATED_CURRENT_A,
    STALL_CURRENT_BASE,
    STALL_CURRENT_MAX_SCORE,
    STALL_CURRENT_SCORE_PER_AMP,
    STALL_SLIP_MAX_SCORE,
    STALL_SLIP_SCORE_PER_PCT,
    STALL_SLIP_THRESHOLD,
)
from physics import PhysicsRecord

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
RuleLevel     = Literal["NORMAL", "WARNING", "DANGER", "CRITICAL"]
RuleFaultType = Literal["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR"]

# ---------------------------------------------------------------------------
# 風險等級對應分數區間（從 config.py 的 LEVEL_THRESHOLDS）
# ---------------------------------------------------------------------------
LEVEL_THRESHOLDS: List[Tuple[int, RuleLevel]] = [
    (75, "CRITICAL"),
    (50, "DANGER"),
    (25, "WARNING"),
    (0,  "NORMAL"),
]

# ---------------------------------------------------------------------------
# 判斷結果
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    fault_type:    RuleFaultType
    level:         RuleLevel
    anomaly_score: int
    contributions: Dict[str, int]
    reasons:       List[str] = field(default_factory=list)


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
# 各條件分數計算
# ---------------------------------------------------------------------------

def _score_stall(current_a: float, slip_ratio: float) -> Tuple[int, List[str]]:
    """
    STALL 判斷：只看電流是否超過 133% FLA。
    滑差條件已移除，因為 slip_ratio 是從電流估算的，
    加入滑差是假的雙重確認，兩個條件不獨立。
    模糊地帶（OVERLOAD vs STALL）由 ML 負責區分。
    """
    score   = 0
    reasons = []

    current_over  = max(0.0, current_a - STALL_CURRENT_BASE)
    current_score = min(int(current_over * STALL_CURRENT_SCORE_PER_AMP), STALL_CURRENT_MAX_SCORE + STALL_SLIP_MAX_SCORE)

    if current_a > STALL_CURRENT_BASE:
        score = current_score
        reasons.append(f"電流 {current_a:.1f}A > {STALL_CURRENT_BASE:.1f}A (133% FLA) [{score}分]")

    return score, reasons


def _score_load_loss(frequency_hz: float, current_a: float) -> Tuple[int, List[str]]:
    """
    LOAD_LOSS 判斷：只看電流是否極低（< 5A）。
    頻率限制已移除，因為任何操作頻率下都可能發生負載流失。
    電流越低分數越高，上限 DANGER 區間。
    """
    score   = 0
    reasons = []

    if current_a < LOAD_LOSS_CURRENT_MAX:
        deficit = max(0.0, LOAD_LOSS_CURRENT_MAX - current_a)
        score   = min(int(deficit / LOAD_LOSS_CURRENT_MAX * 49) + 25, LOAD_LOSS_MAX_SCORE)
        reasons.append(f"電流 {current_a:.1f}A < {LOAD_LOSS_CURRENT_MAX:.0f}A（負載流失）[{score}分]")

    return score, reasons


def _score_overload(current_a: float) -> Tuple[int, List[str]]:
    """
    OVERLOAD 判斷：電流超過 110% FLA。
    上限 WARNING 區間，不讓 OVERLOAD 跑到 DANGER。
    """
    score   = 0
    reasons = []

    over = max(0.0, current_a - OVERLOAD_CURRENT_BASE)
    if over > 0:
        score = min(int(over * OVERLOAD_SCORE_PER_AMP), OVERLOAD_MAX_SCORE)
        reasons.append(f"電流 {current_a:.1f}A > {OVERLOAD_CURRENT_BASE:.1f}A (110% FLA) [{score}分]")

    return score, reasons


def _score_bearing(motor_id: str, current_a: float, frequency_hz: float) -> Tuple[int, List[str]]:
    """
    BEARING_WEAR 判斷：電流波動大且頻率穩定。
    需累積足夠筆數（滑動窗口）才觸發。
    """
    score   = 0
    reasons = []

    window = _get_window(motor_id)
    window.push(current_a, frequency_hz)

    if not window.ready():
        return 0, []

    std        = window.current_std()
    freq_range = window.freq_range()

    if std > BEARING_STD_THRESHOLD and freq_range < BEARING_FREQ_STABLE_RANGE:
        over  = std - BEARING_STD_THRESHOLD
        score = min(int(over * 15) + 20, BEARING_MAX_SCORE)
        reasons.append(f"電流波動標準差 {std:.2f}A > {BEARING_STD_THRESHOLD}A [{score}分]")
        reasons.append(f"頻率穩定（變化 {freq_range:.2f}Hz）[輔助確認]")

    return score, reasons


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _score_to_level(score: int) -> RuleLevel:
    for threshold, level in LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "NORMAL"


def _dominant_fault(contributions: Dict[str, int]) -> RuleFaultType:
    if not contributions:
        return "NORMAL"
    dominant = max(contributions, key=lambda k: contributions[k])
    mapping: Dict[str, RuleFaultType] = {
        "stall":        "STALL",
        "load_loss":    "LOAD_LOSS",
        "overload":     "OVERLOAD",
        "bearing_wear": "BEARING_WEAR",
    }
    return mapping.get(dominant, "NORMAL")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def evaluate(record: PhysicsRecord) -> RuleResult:
    """
    接收 physics.py 的完整物理量，計算 Anomaly Score 並回傳 RuleResult。

    注意：
        啟動遮蔽由 main.py 負責，rules.py 不判斷狀態。
        直接傳入資料就給出判斷結果。
    """
    contributions: Dict[str, int] = {}
    all_reasons:   List[str]      = []

    stall_score,    stall_reasons    = _score_stall(record.current_a, record.slip_ratio)
    load_score,     load_reasons     = _score_load_loss(record.frequency_hz, record.current_a)
    bearing_score,  bearing_reasons  = _score_bearing(
        record.motor_id, record.current_a, record.frequency_hz
    )

    overload_score, overload_reasons = _score_overload(record.current_a)

    # 各條件獨立計分，取最高分那個
    # STALL 和 OVERLOAD 可能同時觸發（電流 20~22A 模糊地帶）
    # 取分數最高的那個，不累加
    all_contributions = {
        "stall":        stall_score,
        "load_loss":    load_score,
        "overload":     overload_score,
        "bearing_wear": bearing_score,
    }

    # 只保留分數最高的那個
    if any(v > 0 for v in all_contributions.values()):
        dominant = max(all_contributions, key=lambda k: all_contributions[k])
        contributions[dominant] = all_contributions[dominant]
        if dominant == "stall":        all_reasons.extend(stall_reasons)
        elif dominant == "load_loss":  all_reasons.extend(load_reasons)
        elif dominant == "overload":   all_reasons.extend(overload_reasons)
        elif dominant == "bearing_wear": all_reasons.extend(bearing_reasons)

    total_score = min(max(all_contributions.values(), default=0), 100)
    level       = _score_to_level(total_score)
    fault_type  = _dominant_fault(contributions) if contributions else "NORMAL"

    if not all_reasons:
        all_reasons = ["各項數值正常"]

    return RuleResult(
        fault_type    = fault_type,
        level         = level,
        anomaly_score = total_score,
        contributions = contributions,
        reasons       = all_reasons,
    )
