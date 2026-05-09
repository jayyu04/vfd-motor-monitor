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

from device_adapter import AdaptedRecord

# ---------------------------------------------------------------------------
# 型別定義
# ---------------------------------------------------------------------------
MlFaultType = Literal["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR"]
MlLevel = Literal["NORMAL", "WARNING", "DANGER", "CRITICAL"]

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
RATED_CURRENT = 15.0
MODEL_PATH = "rf_model.pkl"
ENCODER_PATH = "rf_encoder.pkl"
SAMPLES_PER_CLASS = 1000  # 每個 fault_type 產生的訓練筆數
CURRENT_WINDOW = 5  # 計算電流標準差的滑動窗口大小

# fault_type → 風險等級對應
FAULT_TO_LEVEL: Dict[str, MlLevel] = {
    "NORMAL": "NORMAL",
    "OVERLOAD": "WARNING",
    "BEARING_WEAR": "WARNING",
    "LOAD_LOSS": "DANGER",
    "STALL": "CRITICAL",
}


# ---------------------------------------------------------------------------
# 推論結果
# ---------------------------------------------------------------------------
@dataclass
class MlResult:
    fault_type: MlFaultType
    level: MlLevel
    confidence: float  # 0.0 ~ 1.0
    probabilities: Dict[str, float]  # 每個類別的機率


# ---------------------------------------------------------------------------
# 特徵工程
# ---------------------------------------------------------------------------


def extract_features(record: AdaptedRecord, current_std: float = 0.0) -> List[float]:
    """
    從 AdaptedRecord 提取特徵向量。

    特徵清單：
    1. frequency_hz     頻率
    2. current_a        電流
    3. current_ratio    電流 / 額定電流（正規化）
    4. slip_ratio       轉差率
    5. rpm_est          實際轉速
    6. torque_nm        轉矩
    7. input_power_kw   輸入功率
    8. current_std_5    最近5筆電流標準差（BEARING_WEAR 關鍵）
    """
    return [
        record.frequency_hz,
        record.current_a,
        record.current_a / RATED_CURRENT,  # current_ratio
        record.slip_ratio,
        record.rpm_est,
        record.torque_nm,
        record.input_power_kw,
        current_std,
    ]


FEATURE_NAMES = [
    "frequency_hz",
    "current_a",
    "current_ratio",
    "slip_ratio",
    "rpm_est",
    "torque_nm",
    "input_power_kw",
    "current_std_5",
]

# ---------------------------------------------------------------------------
# 訓練資料產生
# ---------------------------------------------------------------------------


def generate_training_data(samples_per_class: int = SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    用 simulator + device_adapter 批次產生訓練資料。
    每個 fault_type 產生 samples_per_class 筆，共 5 個類別。
    """
    import random
    from collections import deque
    from simulator import generate_sensor_record, _reset_state
    from device_adapter import adapt

    fault_types: List[MlFaultType] = [
        "NORMAL",
        "OVERLOAD",
        "STALL",
        "LOAD_LOSS",
        "BEARING_WEAR",
    ]

    all_rows = []

    for fault_type in fault_types:
        print(f"  產生 {fault_type} 資料 {samples_per_class} 筆...")

        _reset_state("MOTOR-001")
        current_window: Deque[float] = deque(maxlen=CURRENT_WINDOW)

        for _ in range(samples_per_class):
            raw = generate_sensor_record(
                power_state="ON",
                fault_type=fault_type,
                startup_elapsed_sec=10.0,
            )
            rec = adapt(raw)

            current_window.append(rec.current_a)

            # NORMAL 強制低標準差，BEARING_WEAR 強制高標準差
            # 讓模型學到兩者的關鍵差異
            if fault_type == "NORMAL":
                std = (
                    float(np.std(list(current_window)))
                    if len(current_window) > 1
                    else 0.0
                )
                std = min(std, 0.3)  # 壓低，確保 NORMAL 標準差很小
            elif fault_type == "BEARING_WEAR":
                std = (
                    float(np.std(list(current_window)))
                    if len(current_window) > 1
                    else 0.0
                )
                std = max(std, 2.0)  # 拉高，確保 BEARING_WEAR 標準差明顯大
            else:
                std = (
                    float(np.std(list(current_window)))
                    if len(current_window) > 1
                    else 0.0
                )

            features = extract_features(rec, current_std=std)
            row = dict(zip(FEATURE_NAMES, features))
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
    samples_per_class: int = SAMPLES_PER_CLASS, save: bool = True
) -> Tuple[RandomForestClassifier, LabelEncoder]:
    """
    產生訓練資料 → 訓練隨機森林 → 印出報告 → 儲存模型
    """
    print("=" * 60)
    print("開始產生訓練資料...")
    df = generate_training_data(samples_per_class)

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\n開始訓練隨機森林...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 評估
    y_pred = clf.predict(X_test)
    print("\n模型評估報告：")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 特徵重要性
    importances = clf.feature_importances_
    print("特徵重要性：")
    for name, imp in sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
    ):
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

    def __init__(self, model_path: str = MODEL_PATH, encoder_path: str = ENCODER_PATH):
        self._clf: Optional[RandomForestClassifier] = None
        self._le: Optional[LabelEncoder] = None
        self._windows: Dict[str, Deque[float]] = {}
        self._model_path = model_path
        self._encoder_path = encoder_path

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

    def predict(self, record: AdaptedRecord) -> MlResult:
        """
        對單筆 AdaptedRecord 進行推論。
        STARTUP 和 OFF 狀態直接回傳 NORMAL，不進模型。
        """
        assert (
            self._clf is not None and self._le is not None
        ), "請先呼叫 load() 載入模型"

        # 關機或啟動中不推論
        if record.power_state == "OFF" or record.machine_state == "STARTUP":
            return MlResult(
                fault_type="NORMAL",
                level="NORMAL",
                confidence=1.0,
                probabilities={ft: 0.0 for ft in FAULT_TO_LEVEL},
            )

        # 更新電流窗口
        window = self._get_window(record.motor_id)
        window.append(record.current_a)
        raw_std = float(np.std(list(window))) if len(window) > 1 else 0.0

        # 提取特徵並推論（std 直接用原始值）
        features = np.array([extract_features(record, current_std=raw_std)])
        proba = self._clf.predict_proba(features)[0]

        # 組成機率字典
        classes = self._le.classes_
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        # 取最高機率的類別
        best_idx = int(np.argmax(proba))
        fault_type = str(classes[best_idx])
        confidence = float(proba[best_idx])
        level = FAULT_TO_LEVEL.get(fault_type, "NORMAL")

        return MlResult(
            fault_type=fault_type,  # type: ignore[arg-type]
            level=level,
            confidence=round(confidence, 4),
            probabilities=proba_dict,
        )


# ---------------------------------------------------------------------------
# 全域推論器（main.py 直接使用）
# ---------------------------------------------------------------------------

_predictor: Optional[MotorMLPredictor] = None


def load_model() -> None:
    """初始化並載入模型，main.py 啟動時呼叫一次。"""
    global _predictor
    _predictor = MotorMLPredictor()
    _predictor.load()


def predict(record: AdaptedRecord) -> MlResult:
    """對外推論入口，main.py 每筆資料呼叫這個。"""
    global _predictor
    if _predictor is None:
        load_model()
    return _predictor.predict(record)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo() -> None:
    import random
    from simulator import generate_sensor_record, _reset_state
    from device_adapter import adapt

    random.seed(None)

    # 訓練（或載入已有模型）
    load_model()

    print("\n" + "=" * 60)
    print("推論測試")
    print("=" * 60)

    test_cases = [
        ("ON", "NORMAL", 4.0),
        ("ON", "OVERLOAD", 4.0),
        ("ON", "STALL", 4.0),
        ("ON", "LOAD_LOSS", 4.0),
        ("ON", "BEARING_WEAR", 4.0),
    ]

    for power_state, fault_type, elapsed in test_cases:
        _reset_state("MOTOR-001")
        # 先跑幾筆建立窗口
        for _ in range(CURRENT_WINDOW):
            raw = generate_sensor_record(power_state, fault_type, elapsed)
            adapt(raw)

        raw = generate_sensor_record(power_state, fault_type, elapsed)
        rec = adapt(raw)
        res = predict(rec)

        print(
            f"fault_type={fault_type:<14} "
            f"→ ml={res.fault_type:<14} "
            f"level={res.level:<8} "
            f"confidence={res.confidence:.2%}"
        )


if __name__ == "__main__":
    demo()
