from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Generator, List, Optional

from main import FullRecord

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
DB_PATH = "motor_monitor.db"
MAX_HISTORY = 10_000  # 最多保留幾筆（超過自動刪最舊的）

# ---------------------------------------------------------------------------
# 資料庫初始化
# ---------------------------------------------------------------------------


def init_db(db_path: str = DB_PATH) -> None:
    """建立資料表（若已存在則跳過）"""
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS motor_records (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                motor_id         TEXT    NOT NULL,
                power_state      TEXT    NOT NULL,
                machine_state    TEXT    NOT NULL,
                fault_type       TEXT    NOT NULL,

                -- 電氣 / 機械量
                frequency_hz     REAL,
                current_a        REAL,
                voltage_v        REAL,
                sync_rpm         REAL,
                slip_ratio       REAL,
                rpm_est          REAL,
                power_factor     REAL,
                input_power_kw   REAL,
                output_power_kw  REAL,
                torque_nm        REAL,

                -- rules 結果
                rule_fault_type  TEXT,
                rule_level       TEXT,
                rule_score       INTEGER,
                rule_reasons     TEXT,    -- JSON array
                rule_is_startup  INTEGER, -- 0/1

                -- ml 結果
                ml_fault_type    TEXT,
                ml_level         TEXT,
                ml_confidence    REAL,
                ml_probabilities TEXT,    -- JSON object

                -- 綜合
                final_level      TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON motor_records (timestamp DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_motor_id
            ON motor_records (motor_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_final_level
            ON motor_records (final_level)
        """)


# ---------------------------------------------------------------------------
# 連線工具
# ---------------------------------------------------------------------------


@contextmanager
def _connect(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 寫入
# ---------------------------------------------------------------------------


def insert_record(record: FullRecord, db_path: str = DB_PATH) -> int:
    """
    寫入一筆 FullRecord，回傳新插入的 id。
    超過 MAX_HISTORY 時自動刪除最舊的資料。
    """
    d = asdict(record)

    # List / Dict 轉 JSON 字串儲存
    d["rule_reasons"] = json.dumps(d["rule_reasons"], ensure_ascii=False)
    d["ml_probabilities"] = json.dumps(d["ml_probabilities"], ensure_ascii=False)
    d["rule_is_startup"] = int(d["rule_is_startup"])

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO motor_records (
                timestamp, motor_id, power_state, machine_state, fault_type,
                frequency_hz, current_a, voltage_v,
                sync_rpm, slip_ratio, rpm_est,
                power_factor, input_power_kw, output_power_kw, torque_nm,
                rule_fault_type, rule_level, rule_score, rule_reasons, rule_is_startup,
                ml_fault_type, ml_level, ml_confidence, ml_probabilities,
                final_level
            ) VALUES (
                :timestamp, :motor_id, :power_state, :machine_state, :fault_type,
                :frequency_hz, :current_a, :voltage_v,
                :sync_rpm, :slip_ratio, :rpm_est,
                :power_factor, :input_power_kw, :output_power_kw, :torque_nm,
                :rule_fault_type, :rule_level, :rule_score, :rule_reasons, :rule_is_startup,
                :ml_fault_type, :ml_level, :ml_confidence, :ml_probabilities,
                :final_level
            )
        """,
            d,
        )

        new_id = cursor.lastrowid

        # 超過上限時刪最舊的
        count = conn.execute("SELECT COUNT(*) FROM motor_records").fetchone()[0]
        if count > MAX_HISTORY:
            conn.execute(
                """
                DELETE FROM motor_records
                WHERE id IN (
                    SELECT id FROM motor_records
                    ORDER BY id ASC
                    LIMIT ?
                )
            """,
                (count - MAX_HISTORY,),
            )

    return new_id


# ---------------------------------------------------------------------------
# 讀取
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row) -> Dict:
    d = dict(row)
    d["rule_reasons"] = json.loads(d["rule_reasons"])
    d["ml_probabilities"] = json.loads(d["ml_probabilities"])
    d["rule_is_startup"] = bool(d["rule_is_startup"])
    return d


def fetch_latest(
    n: int = 50, motor_id: Optional[str] = None, db_path: str = DB_PATH
) -> List[Dict]:
    """取最新 n 筆，可依 motor_id 篩選"""
    with _connect(db_path) as conn:
        if motor_id:
            rows = conn.execute(
                """
                SELECT * FROM motor_records
                WHERE motor_id = ?
                ORDER BY id DESC LIMIT ?
            """,
                (motor_id, n),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM motor_records
                ORDER BY id DESC LIMIT ?
            """,
                (n,),
            ).fetchall()

    return [_row_to_dict(r) for r in rows]


def fetch_alerts(
    level: str = "WARNING", n: int = 100, db_path: str = DB_PATH
) -> List[Dict]:
    """
    取出風險等級 >= level 的告警紀錄。
    level 可傳入 WARNING / DANGER / CRITICAL。
    """
    level_map = {"WARNING": 1, "DANGER": 2, "CRITICAL": 3}
    min_order = level_map.get(level, 1)
    levels = [k for k, v in level_map.items() if v >= min_order]
    placeholders = ",".join("?" * len(levels))

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM motor_records
            WHERE final_level IN ({placeholders})
            ORDER BY id DESC LIMIT ?
        """,
            (*levels, n),
        ).fetchall()

    return [_row_to_dict(r) for r in rows]


def fetch_stats(motor_id: Optional[str] = None, db_path: str = DB_PATH) -> Dict:
    """
    取統計摘要：
    - 總筆數
    - 各 final_level 分佈
    - 各 fault_type 分佈
    - 最新一筆時間
    """
    where = "WHERE motor_id = ?" if motor_id else ""
    params = (motor_id,) if motor_id else ()

    with _connect(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM motor_records {where}", params
        ).fetchone()[0]

        level_rows = conn.execute(
            f"SELECT final_level, COUNT(*) as cnt FROM motor_records {where} "
            f"GROUP BY final_level",
            params,
        ).fetchall()

        fault_rows = conn.execute(
            f"SELECT rule_fault_type, COUNT(*) as cnt FROM motor_records {where} "
            f"GROUP BY rule_fault_type",
            params,
        ).fetchall()

        latest_ts = conn.execute(
            f"SELECT timestamp FROM motor_records {where} " f"ORDER BY id DESC LIMIT 1",
            params,
        ).fetchone()

    return {
        "total": total,
        "level_dist": {r["final_level"]: r["cnt"] for r in level_rows},
        "fault_dist": {r["rule_fault_type"]: r["cnt"] for r in fault_rows},
        "latest_timestamp": latest_ts[0] if latest_ts else None,
    }


def clear_all(db_path: str = DB_PATH) -> None:
    """清空所有資料（測試用）"""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM motor_records")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo() -> None:
    import random
    import os

    random.seed(None)

    from ml_model import load_model
    from main import MotorMonitor

    # 清掉舊的測試資料庫
    if os.path.exists("test_motor.db"):
        os.remove("test_motor.db")

    db = "test_motor.db"
    init_db(db)
    print("資料庫初始化完成")

    load_model()
    monitor = MotorMonitor(interval_sec=1.0)

    print("\n寫入各模式資料各 5 筆...")
    from simulator import FaultType

    fault_types: List[FaultType] = [
        "NORMAL",
        "OVERLOAD",
        "STALL",
        "LOAD_LOSS",
        "BEARING_WEAR",
    ]

    for fault_type in fault_types:
        monitor.set_power("ON")
        monitor.set_fault_type(fault_type)
        monitor._state.startup_elapsed_sec = 10.0
        for _ in range(5):
            rec = monitor.tick()
            insert_record(rec, db_path=db)

    print("\n最新 5 筆：")
    for row in fetch_latest(5, db_path=db):
        print(
            f"  {row['timestamp']}  "
            f"fault={row['fault_type']:<14} "
            f"final={row['final_level']}"
        )

    print("\n告警紀錄（>= WARNING）：")
    alerts = fetch_alerts("WARNING", db_path=db)
    print(f"  共 {len(alerts)} 筆告警")
    for row in alerts[:5]:
        print(
            f"  {row['timestamp']}  "
            f"rule={row['rule_fault_type']:<14} "
            f"ml={row['ml_fault_type']:<14} "
            f"final={row['final_level']}"
        )

    print("\n統計摘要：")
    stats = fetch_stats(db_path=db)
    print(f"  總筆數：{stats['total']}")
    print(f"  等級分佈：{stats['level_dist']}")
    print(f"  異常分佈：{stats['fault_dist']}")
    print(f"  最新時間：{stats['latest_timestamp']}")


if __name__ == "__main__":
    demo()
