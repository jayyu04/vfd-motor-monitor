from __future__ import annotations

import os
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    RATED_CURRENT_A,
    BEARING_WINDOW_SIZE,
)
from physics import PhysicsRecord

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
MlFaultType = Literal["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR"]
MlLevel     = Literal["NORMAL", "WARNING", "DANGER", "CRITICAL"]

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
MODEL_PATH        = "rf_model.pkl"
ENCODER_PATH      = "rf_encoder.pkl"
SAMPLES_PER_CLASS = 1000
CURRENT_WINDOW    = BEARING_WINDOW_SIZE

# fault_type → 風險等級對應
# STALL 是漸進式，ML 無法知道電流確切值來判斷 DANGER 或 CRITICAL
# 所以 ML 輸出 STALL 時統一給 DANGER，讓 Rules 的漸進分數決定最終等級
FAULT_TO_LEVEL: Dict[str, MlLevel] = {
    "NORMAL":       "NORMAL",
    "OVERLOAD":     "WARNING",
    "BEARING_WEAR": "WARNING",
    "LOAD_LOSS":    "DANGER",
    "STALL":        "DANGER",
}

# ---------------------------------------------------------------------------
# 推論結果
# ---------------------------------------------------------------------------
@dataclass
class MlResult:
    fault_type:    MlFaultType
    level:         MlLevel
    confidence:    float
    probabilities: Dict[str, float]


# ---------------------------------------------------------------------------
# 特徵工程（6個特徵）
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "frequency_hz",
    "current_a",
    "current_ratio",
    "slip_ratio",
    "torque_nm",
    "current_std",   # 滑動窗口標準差，窗口大小由 config.BEARING_WINDOW_SIZE 決定（目前 20 筆）
]


def extract_features(record: PhysicsRecord, current_std: float = 0.0) -> List[float]:
    """
    從 PhysicsRecord 提取 6 個特徵向量。

    移除的特徵：
        rpm_est       → 從電流估算，跟 current_a 高度相關，資訊重複
        input_power_kw → power_factor 是估算值，準確性有限

    保留的特徵：
        frequency_hz   → 直接感測器，頻率狀態
        current_a      → 直接感測器，電流絕對值
        current_ratio  → 正規化電流，相對負載比
        slip_ratio     → 負載狀態，STALL 關鍵特徵
        torque_nm      → 最重要的綜合指標
        current_std_5  → BEARING_WEAR 唯一關鍵特徵
    """
    return [
        record.frequency_hz,
        record.current_a,
        record.current_a / RATED_CURRENT_A,   # current_ratio
        record.slip_ratio,
        record.torque_nm,
        current_std,
    ]


# ---------------------------------------------------------------------------
# 訓練資料產生
# ---------------------------------------------------------------------------

def generate_training_data(samples_per_class: int = SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    用新架構（VFD_simulator → comms → data_collector → physics）
    批次產生訓練資料。
    每個 fault_type 產生 samples_per_class 筆，共 5 個類別。
    """
    from comms import VFDComms
    from data_collector import collect
    from physics import calculate
    from VFD_simulator import _reset_all_states

    fault_types: List[MlFaultType] = [
        "NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR"
    ]

    comms    = VFDComms(mode="MOCK")
    all_rows = []

    for fault_type in fault_types:
        print(f"  產生 {fault_type} 資料 {samples_per_class} 筆...")

        current_window: Deque[float] = deque(maxlen=CURRENT_WINDOW)

        # BEARING_WEAR 先跑預熱，讓窗口填滿，確保訓練資料的 current_std 有意義
        if fault_type == "BEARING_WEAR":
            for _ in range(CURRENT_WINDOW):
                raw = comms.read(power_state="ON", fault_type=fault_type, elapsed_sec=10.0)
                rec = collect(raw)
                phy = calculate(rec)
                current_window.append(phy.current_a)

        for i in range(samples_per_class):
            # 每個 fault_type 第一筆重置狀態（BEARING_WEAR 已在上方預熱）
            if i == 0 and fault_type != "BEARING_WEAR":
                _reset_all_states()
            # NORMAL 每 20 筆重置一次（跟窗口大小一致）確保頻率覆蓋 30~60Hz 全範圍
            elif fault_type == "NORMAL" and i % 20 == 0:
                _reset_all_states()
                current_window.clear()

            raw = comms.read(
                power_state = "ON",
                fault_type  = fault_type,
                elapsed_sec = 10.0,
            )
            rec = collect(raw)
            phy = calculate(rec)

            current_window.append(phy.current_a)

            # 直接用真實的電流標準差，不強制調整
            std = float(np.std(list(current_window))) if len(current_window) > 1 else 0.0

            features = extract_features(phy, current_std=std)
            row      = dict(zip(FEATURE_NAMES, features))
            row["label"] = fault_type
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"  訓練資料總筆數：{len(df)}")
    print(f"  各類別分佈：\n{df['label'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# 訓練
# ---------------------------------------------------------------------------

def train(
    samples_per_class: int = SAMPLES_PER_CLASS,
    save: bool = True,
) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """
    產生訓練資料 → 訓練隨機森林 → 印出報告 → 儲存模型
    """
    print("=" * 60)
    print("開始產生訓練資料...")
    df = generate_training_data(samples_per_class)

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\n開始訓練隨機森林（100棵決策樹）...")
    clf = RandomForestClassifier(
        n_estimators   = 100,
        max_depth      = None,
        min_samples_split = 5,
        random_state   = 42,
        n_jobs         = -1,
    )
    clf.fit(X_train, y_train)

    # 評估
    y_pred = clf.predict(X_test)
    print("\n模型評估報告：")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 特徵重要性
    importances = clf.feature_importances_
    print("特徵重要性：")
    for name, imp in sorted(zip(FEATURE_NAMES, importances),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"  {name:<18} {imp:.4f}  {bar}")

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(le, f)
        print(f"\n模型已儲存：{MODEL_PATH}, {ENCODER_PATH}")

    return clf, le


# ---------------------------------------------------------------------------
# 推論引擎
# ---------------------------------------------------------------------------

class MotorMLPredictor:
    """
    封裝模型載入與推論邏輯。
    維護每台馬達的電流滑動窗口，確保 current_std_5 特徵正確計算。
    """

    def __init__(
        self,
        model_path:   str = MODEL_PATH,
        encoder_path: str = ENCODER_PATH,
    ):
        self._clf:          Optional[RandomForestClassifier] = None
        self._le:           Optional[LabelEncoder]           = None
        self._windows:      Dict[str, Deque[float]]          = {}
        self._model_path    = model_path
        self._encoder_path  = encoder_path

    def load(self) -> None:
        """載入已訓練的模型。若模型不存在則自動訓練。"""
        if not os.path.exists(self._model_path):
            print("找不到模型檔案，開始自動訓練...")
            self._clf, self._le = train(save=True)
        else:
            with open(self._model_path, "rb") as f:
                self._clf = pickle.load(f)
            with open(self._encoder_path, "rb") as f:
                self._le = pickle.load(f)
            print(f"模型載入成功：{self._model_path}")

    def _get_window(self, motor_id: str) -> Deque[float]:
        if motor_id not in self._windows:
            self._windows[motor_id] = deque(maxlen=CURRENT_WINDOW)
        return self._windows[motor_id]

    def clear_windows(self) -> None:
        """關機時清除所有電流窗口。"""
        self._windows.clear()

    def predict(self, record: PhysicsRecord) -> MlResult:
        """
        對單筆 PhysicsRecord 進行推論。

        注意：
            啟動遮蔽由 main.py 負責。
            ml_model 只負責推論，不判斷狀態。
        """
        assert self._clf is not None and self._le is not None, \
            "請先呼叫 load() 載入模型"

        # 更新電流窗口
        window  = self._get_window(record.motor_id)
        window.append(record.current_a)
        raw_std = float(np.std(list(window))) if len(window) > 1 else 0.0

        # 提取特徵並推論
        features = np.array([extract_features(record, current_std=raw_std)])
        proba    = self._clf.predict_proba(features)[0]

        # 組成機率字典
        classes    = self._le.classes_
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        # 取最高機率的類別
        best_idx   = int(np.argmax(proba))
        fault_type = str(classes[best_idx])
        confidence = float(proba[best_idx])
        level      = FAULT_TO_LEVEL.get(fault_type, "NORMAL")

        return MlResult(
            fault_type    = fault_type,     # type: ignore[arg-type]
            level         = level,
            confidence    = round(confidence, 4),
            probabilities = proba_dict,
        )


# ---------------------------------------------------------------------------
# 全域推論器
# ---------------------------------------------------------------------------

_predictor: Optional[MotorMLPredictor] = None


def load_model() -> None:
    """初始化並載入模型，系統啟動時呼叫一次。"""
    global _predictor
    _predictor = MotorMLPredictor()
    _predictor.load()


def predict(record: PhysicsRecord) -> MlResult:
    """對外推論入口，main.py 每筆資料呼叫這個。"""
    global _predictor
    if _predictor is None:
        load_model()
    return _predictor.predict(record)  # type: ignore[union-attr]


def clear_ml_windows() -> None:
    """關機時清除電流窗口。"""
    global _predictor
    if _predictor is not None:
        _predictor.clear_windows()
