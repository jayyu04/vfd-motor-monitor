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
RuleLevel = Literal["NORMAL", "WARNING", "DANGER", "CRITICAL"]
RuleFaultType = Literal["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR"]

# ---------------------------------------------------------------------------
# 風險等級對應分數區間（從 config.py 的 LEVEL_THRESHOLDS）
# ---------------------------------------------------------------------------
LEVEL_THRESHOLDS: List[Tuple[int, RuleLevel]] = [
    (65, "CRITICAL"),
    (40, "DANGER"),
    (15, "WARNING"),
    (0, "NORMAL"),
]


# ---------------------------------------------------------------------------
# 判斷結果
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    fault_type: RuleFaultType
    level: RuleLevel
    anomaly_score: int
    contributions: Dict[str, int]
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 滑動窗口（BEARING_WEAR 用）
# ---------------------------------------------------------------------------
class _SlidingWindow:
    def __init__(self, size: int):
        self._size = size
        self._current: List[float] = []
        self._freq: List[float] = []

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
        mean = sum(self._current) / len(self._current)
        variance = sum((x - mean) ** 2 for x in self._current) / len(self._current)
        return variance**0.5

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
    STALL 判斷：電流和滑差雙條件確認。
    兩個條件都滿足才給滿分，單一條件只給一半。
    """
    score = 0
    reasons = []

    current_over = max(0.0, current_a - STALL_CURRENT_BASE)
    slip_over = max(0.0, slip_ratio - STALL_SLIP_THRESHOLD)
    current_score = min(
        int(current_over * STALL_CURRENT_SCORE_PER_AMP), STALL_CURRENT_MAX_SCORE
    )
    slip_score = min(
        int(slip_over * 600 * STALL_SLIP_SCORE_PER_PCT / 6), STALL_SLIP_MAX_SCORE
    )
    both = current_a > STALL_CURRENT_BASE and slip_ratio > STALL_SLIP_THRESHOLD

    if current_a > STALL_CURRENT_BASE:
        s = current_score if both else current_score // 2
        score += s
        reasons.append(
            f"電流 {current_a:.1f}A > {STALL_CURRENT_BASE:.1f}A (133% FLA) [{s}分]"
        )

    if slip_ratio > STALL_SLIP_THRESHOLD:
        s = slip_score if both else slip_score // 2
        score += s
        reasons.append(f"轉差率 {slip_ratio:.1%} > {STALL_SLIP_THRESHOLD:.0%} [{s}分]")

    return score, reasons


def _score_load_loss(frequency_hz: float, current_a: float) -> Tuple[int, List[str]]:
    """
    LOAD_LOSS 判斷：頻率正常但電流極低。
    兩個條件都滿足才給分，電流越低分數越高。
    """
    score = 0
    reasons = []

    if frequency_hz > LOAD_LOSS_FREQ_MIN and current_a < LOAD_LOSS_CURRENT_MAX:
        deficit = max(0.0, LOAD_LOSS_CURRENT_MAX - current_a)
        score = min(int(deficit / LOAD_LOSS_CURRENT_MAX * 30) + 40, LOAD_LOSS_MAX_SCORE)
        half = score // 2
        reasons.append(
            f"頻率 {frequency_hz:.1f}Hz > {LOAD_LOSS_FREQ_MIN:.0f}Hz [{half}分]"
        )
        reasons.append(
            f"電流 {current_a:.1f}A < {LOAD_LOSS_CURRENT_MAX:.0f}A（負載流失）[{half}分]"
        )

    return score, reasons


def _score_overload(current_a: float) -> Tuple[int, List[str]]:
    """
    OVERLOAD 判斷：電流超過 110% FLA。
    上限 WARNING 區間，不讓 OVERLOAD 跑到 DANGER。
    """
    score = 0
    reasons = []

    over = max(0.0, current_a - OVERLOAD_CURRENT_BASE)
    if over > 0:
        score = min(int(over * OVERLOAD_SCORE_PER_AMP), OVERLOAD_MAX_SCORE)
        reasons.append(
            f"電流 {current_a:.1f}A > {OVERLOAD_CURRENT_BASE:.1f}A (110% FLA) [{score}分]"
        )

    return score, reasons


def _score_bearing(
    motor_id: str, current_a: float, frequency_hz: float
) -> Tuple[int, List[str]]:
    """
    BEARING_WEAR 判斷：電流波動大且頻率穩定。
    需累積足夠筆數（滑動窗口）才觸發。
    """
    score = 0
    reasons = []

    window = _get_window(motor_id)
    window.push(current_a, frequency_hz)

    if not window.ready():
        return 0, []

    std = window.current_std()
    freq_range = window.freq_range()

    if std > BEARING_STD_THRESHOLD and freq_range < BEARING_FREQ_STABLE_RANGE:
        over = std - BEARING_STD_THRESHOLD
        score = min(int(over * 15) + 20, BEARING_MAX_SCORE)
        reasons.append(
            f"電流波動標準差 {std:.2f}A > {BEARING_STD_THRESHOLD}A [{score}分]"
        )
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
        "stall": "STALL",
        "load_loss": "LOAD_LOSS",
        "overload": "OVERLOAD",
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
    all_reasons: List[str] = []

    stall_score, stall_reasons = _score_stall(record.current_a, record.slip_ratio)
    load_score, load_reasons = _score_load_loss(record.frequency_hz, record.current_a)
    bearing_score, bearing_reasons = _score_bearing(
        record.motor_id, record.current_a, record.frequency_hz
    )

    # OVERLOAD 在 STALL 已高分時不重複計算
    if stall_score < 30:
        overload_score, overload_reasons = _score_overload(record.current_a)
    else:
        overload_score, overload_reasons = 0, []

    if stall_score > 0:
        contributions["stall"] = stall_score
        all_reasons.extend(stall_reasons)
    if load_score > 0:
        contributions["load_loss"] = load_score
        all_reasons.extend(load_reasons)
    if overload_score > 0:
        contributions["overload"] = overload_score
        all_reasons.extend(overload_reasons)
    if bearing_score > 0:
        contributions["bearing_wear"] = bearing_score
        all_reasons.extend(bearing_reasons)

    total_score = min(sum(contributions.values()), 100)
    level = _score_to_level(total_score)
    fault_type = _dominant_fault(contributions) if contributions else "NORMAL"

    if not all_reasons:
        all_reasons = ["各項數值正常"]

    return RuleResult(
        fault_type=fault_type,
        level=level,
        anomaly_score=total_score,
        contributions=contributions,
        reasons=all_reasons,
    )

