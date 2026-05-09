from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from device_adapter import AdaptedRecord

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
RuleLevel = Literal["NORMAL", "WARNING", "DANGER", "CRITICAL"]
RuleFaultType = Literal[
    "NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR", "STARTUP"
]

# ---------------------------------------------------------------------------
# 風險等級對應分數區間
# ---------------------------------------------------------------------------
#   0  ~ 29  → NORMAL
#   30 ~ 59  → WARNING
#   60 ~ 79  → DANGER
#   80 ~ 100 → CRITICAL

LEVEL_THRESHOLDS: List[Tuple[int, RuleLevel]] = [
    (65, "CRITICAL"),
    (40, "DANGER"),
    (15, "WARNING"),
    (0, "NORMAL"),
]

# ---------------------------------------------------------------------------
# 馬達規格
# ---------------------------------------------------------------------------
RATED_CURRENT = 15.0  # FLA (A)

# ---------------------------------------------------------------------------
# 各條件閾值
# ---------------------------------------------------------------------------

# STALL
STALL_CURRENT_BASE = 20.0  # 133% FLA（避免與 OVERLOAD 重疊）
STALL_SLIP_BASE = 0.12  # 滑差 12%（OVERLOAD 滑差約 8~10%，拉高門檻）

# LOAD_LOSS
LOAD_LOSS_FREQ_MIN = 40.0
LOAD_LOSS_CURRENT_MAX = 5.0

# OVERLOAD
OVERLOAD_CURRENT_BASE = RATED_CURRENT * 1.10  # 110% FLA = 16.5A

# BEARING_WEAR
BEARING_WINDOW_SIZE = 5
BEARING_STD_THRESHOLD = 0.8
BEARING_FREQ_STABLE_RANGE = 0.5


# ---------------------------------------------------------------------------
# 判斷結果
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    fault_type: RuleFaultType
    level: RuleLevel
    anomaly_score: int
    contributions: Dict[str, int]  # 各條件的分數貢獻
    reasons: List[str]
    is_startup: bool


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


_windows: Dict[str, _SlidingWindow] = {}


def _get_window(motor_id: str) -> _SlidingWindow:
    if motor_id not in _windows:
        _windows[motor_id] = _SlidingWindow(BEARING_WINDOW_SIZE)
    return _windows[motor_id]


# ---------------------------------------------------------------------------
# 各條件分數計算
# ---------------------------------------------------------------------------


def _score_stall(current_a: float, slip_ratio: float) -> Tuple[int, List[str]]:
    """
    兩個條件都滿足才給滿分，單一條件只給一半。
    電流每超過 120% FLA 1A → +8 分（最多 48）
    滑差每超過 10% 一個百分點 → +6 分（最多 36）
    """
    score = 0
    reasons = []

    current_over = max(0.0, current_a - STALL_CURRENT_BASE)
    slip_over = max(0.0, slip_ratio - STALL_SLIP_BASE)
    current_score = min(int(current_over * 8), 48)
    slip_score = min(int(slip_over * 600), 36)
    both = current_a > STALL_CURRENT_BASE and slip_ratio > STALL_SLIP_BASE

    if current_a > STALL_CURRENT_BASE:
        s = current_score if both else current_score // 2
        score += s
        reasons.append(
            f"電流 {current_a:.1f}A > {STALL_CURRENT_BASE:.1f}A (120% FLA) [{s}分]"
        )

    if slip_ratio > STALL_SLIP_BASE:
        s = slip_score if both else slip_score // 2
        score += s
        reasons.append(f"轉差率 {slip_ratio:.1%} > {STALL_SLIP_BASE:.0%} [{s}分]")

    return score, reasons


def _score_load_loss(frequency_hz: float, current_a: float) -> Tuple[int, List[str]]:
    """
    兩個條件都滿足才給分，電流越低分數越高（最多 70）。
    """
    score = 0
    reasons = []

    if frequency_hz > LOAD_LOSS_FREQ_MIN and current_a < LOAD_LOSS_CURRENT_MAX:
        deficit = max(0.0, LOAD_LOSS_CURRENT_MAX - current_a)
        score = min(
            int(deficit / LOAD_LOSS_CURRENT_MAX * 30) + 40, 64
        )  # 上限 64，穩定在 DANGER
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
    電流超過 110% FLA 才給分，每超過 1A → +6 分。
    上限 29 分，確保 OVERLOAD 最高只到 WARNING，不進入 DANGER。
    """
    score = 0
    reasons = []

    over = max(0.0, current_a - OVERLOAD_CURRENT_BASE)
    if over > 0:
        score = min(int(over * 6), 29)  # 上限 29，WARNING 區間最高值
        reasons.append(
            f"電流 {current_a:.1f}A > {OVERLOAD_CURRENT_BASE:.1f}A (110% FLA) [{score}分]"
        )

    return score, reasons


def _score_bearing(
    motor_id: str, current_a: float, frequency_hz: float
) -> Tuple[int, List[str]]:
    """
    累積足夠筆數後，電流標準差 > 閾值 且 頻率穩定才給分（最多 40）。
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
        score = min(int(over * 15) + 20, 39)  # 上限 39，確保只到 WARNING
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


def evaluate(record: AdaptedRecord) -> RuleResult:
    """
    計算 anomaly score（0~100）並回傳 RuleResult。
    各條件分數獨立計算後加總，clamp 到 100。
    fault_type 由分數最高的條件決定。
    """

    if record.power_state == "OFF":
        return RuleResult(
            fault_type="NORMAL",
            level="NORMAL",
            anomaly_score=0,
            contributions={},
            reasons=["馬達已關機"],
            is_startup=False,
        )

    if record.machine_state == "STARTUP":
        return RuleResult(
            fault_type="STARTUP",
            level="NORMAL",
            anomaly_score=0,
            contributions={},
            reasons=["啟動過渡期，暫停異常判斷"],
            is_startup=True,
        )

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
        is_startup=False,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo() -> None:
    import random
    from simulator import generate_sensor_record, _reset_state
    from device_adapter import adapt

    random.seed(None)

    print("=" * 70)
    print("單筆測試")
    print("=" * 70)

    for power_state, fault_type, elapsed in [
        ("OFF", "NORMAL", 0.0),
        ("ON", "NORMAL", 0.0),
        ("ON", "NORMAL", 4.0),
        ("ON", "OVERLOAD", 4.0),
        ("ON", "STALL", 4.0),
        ("ON", "LOAD_LOSS", 4.0),
    ]:
        _reset_state("MOTOR-001")
        _windows.clear()
        raw = generate_sensor_record(power_state, fault_type, elapsed)
        rec = adapt(raw)
        res = evaluate(rec)
        print(
            f"fault_type={fault_type:<14} "
            f"→ rule={res.fault_type:<14} "
            f"score={res.anomaly_score:>3}  "
            f"level={res.level:<8}  "
            f"{res.contributions}"
        )

    print()
    print("=" * 70)
    print("BEARING_WEAR 累積窗口（連續 10 筆）")
    print("=" * 70)

    _reset_state("MOTOR-001")
    _windows.clear()
    for i in range(10):
        raw = generate_sensor_record("ON", "BEARING_WEAR", 4.0)
        rec = adapt(raw)
        res = evaluate(rec)
        print(
            f"  筆 {i+1:>2}  A={rec.current_a:.2f}  "
            f"→ score={res.anomaly_score:>3}  {res.fault_type:<14} {res.level}"
        )

    print()
    print("=" * 70)
    print("分數漸進測試（STALL 從輕微到嚴重）")
    print("=" * 70)

    from device_adapter import AdaptedRecord

    for current in [16.0, 18.5, 21.0, 24.0, 27.0, 30.0]:
        slip = 0.02 + (current / RATED_CURRENT) * 0.08
        r = AdaptedRecord(
            timestamp="2026-01-01T00:00:00",
            motor_id="MOTOR-TEST",
            power_state="ON",
            machine_state="RUNNING",
            fault_type="STALL",
            frequency_hz=45.0,
            current_a=current,
            voltage_v=380.0,
            sync_rpm=1350.0,
            slip_ratio=slip,
            rpm_est=1200.0,
            power_factor=0.85,
            input_power_kw=10.0,
            output_power_kw=8.8,
            torque_nm=70.0,
        )
        _windows.clear()
        res = evaluate(r)
        print(
            f"  A={current:.1f}  slip={slip:.3f}  "
            f"→ score={res.anomaly_score:>3}  {res.level}"
        )


if __name__ == "__main__":
    demo()
