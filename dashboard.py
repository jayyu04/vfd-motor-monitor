from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

from control import get_controller
from database import fetch_latest, fetch_stats


@st.cache_resource
def init_ml_model():
    """ML 模型只載入一次，Streamlit rerun 不重新訓練。"""
    from ml_model import load_model
    load_model()


from config import (
    STARTUP_DURATION_SEC,
    DATA_INTERVAL_MS as DATA_GENERATE_INTERVAL_MS_CFG,
    DASHBOARD_REFRESH_MS,
    MAX_CHART_POINTS as MAX_CHART_POINTS_CFG,
)

# ---------------------------------------------------------------------------
# 控制參數
# ---------------------------------------------------------------------------
DATA_GENERATE_INTERVAL_MS = 300
DASHBOARD_REFRESH_INTERVAL_MS = 500
MAX_CHART_POINTS = 200

FAULT_TYPES = ["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR", "AUTO"]

FAULT_LABEL_MAP = {
    "NORMAL": "正常",
    "OVERLOAD": "過電流",
    "STALL": "機械卡死",
    "LOAD_LOSS": "負載斷裂",
    "BEARING_WEAR": "軸承磨損",
    "STARTUP": "啟動中",
    "AUTO": "自動測試",
}

LEVEL_LABEL_MAP = {
    "NORMAL": "正常",
    "WARNING": "警告",
    "DANGER": "危險",
    "CRITICAL": "緊急",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "is_running": False,
        "selected_fault": "NORMAL",
        "applied_fault": "NORMAL",
        "last_generated_ts": 0.0,
        "page": "dashboard",
        "startup_begin": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def get_ctrl() -> object:
    return get_controller()


def get_startup_elapsed_sec() -> float:
    ctrl = get_controller()
    return ctrl.startup_elapsed_sec


def get_current_machine_state() -> str:
    return get_controller().machine_state


def maybe_generate_data() -> None:
    ctrl = get_controller()
    if not st.session_state.is_running or ctrl.power_state != "ON":
        return

    now = time.time()
    if now - st.session_state.last_generated_ts < DATA_GENERATE_INTERVAL_MS / 1000.0:
        return

    _sb = st.session_state.get("startup_begin", None)
    if _sb is not None:
        ctrl._startup_begin = _sb

    ctrl.set_fault_type(st.session_state.selected_fault)
    ctrl.tick()
    st.session_state.applied_fault = st.session_state.selected_fault
    st.session_state.last_generated_ts = now


def load_data() -> pd.DataFrame:
    rows = fetch_latest(MAX_CHART_POINTS)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp", ascending=True).reset_index(drop=True)


def fmt_fault(v: str) -> str:
    return FAULT_LABEL_MAP.get(str(v), str(v))


def fmt_level(v: str) -> str:
    return LEVEL_LABEL_MAP.get(str(v), str(v))


# ---------------------------------------------------------------------------
# Top control bar
# ---------------------------------------------------------------------------
def render_top_controls() -> None:
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top right, rgba(56,189,248,.05), transparent 28%),
            radial-gradient(circle at top left, rgba(34,211,165,.03), transparent 24%),
            #0a0e14;
    }
    [data-testid="stHeader"] { background: transparent; }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    #MainMenu, footer { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    div[data-testid="stHorizontalBlock"] {
        align-items: end;
        gap: 10px;
    }

    button[data-testid="baseButton-primary"] {
        background: linear-gradient(180deg, rgba(34,211,165,.12), rgba(34,211,165,.06)) !important;
        border: 1px solid rgba(34,211,165,0.42) !important;
        color: #22d3a5 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: .05em !important;
        border-radius: 10px !important;
        box-shadow: 0 0 0 1px rgba(34,211,165,.05) inset, 0 4px 14px rgba(0,0,0,.14) !important;
        transition: all .2s ease !important;
        min-height: 42px !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(180deg, rgba(34,211,165,.20), rgba(34,211,165,.10)) !important;
        border-color: rgba(34,211,165,.72) !important;
        color: #ffffff !important;
        box-shadow: 0 0 18px rgba(34,211,165,.18), 0 4px 18px rgba(0,0,0,.18) !important;
    }

    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(180deg, rgba(248,113,113,.10), rgba(248,113,113,.05)) !important;
        border: 1px solid rgba(248,113,113,0.34) !important;
        color: #f87171 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: .05em !important;
        border-radius: 10px !important;
        box-shadow: 0 0 0 1px rgba(248,113,113,.04) inset, 0 4px 14px rgba(0,0,0,.14) !important;
        transition: all .2s ease !important;
        min-height: 42px !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(180deg, rgba(248,113,113,.18), rgba(248,113,113,.08)) !important;
        border-color: rgba(248,113,113,.68) !important;
        color: #ffffff !important;
        box-shadow: 0 0 18px rgba(248,113,113,.16), 0 4px 18px rgba(0,0,0,.18) !important;
    }

    button[data-testid="baseButton-secondary"]:disabled {
        opacity: .38 !important;
        filter: grayscale(.15);
    }

    div[data-testid="stSelectbox"] > div > div {
        background: linear-gradient(180deg, #111827, #0f1520) !important;
        border: 1px solid rgba(56,189,248,0.20) !important;
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        border-radius: 10px !important;
        min-height: 42px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,.14);
    }

    div[data-testid="stSelectbox"] label {
        color: #64748b !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    iframe { border: none !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    c_on, c_off, c_mode, c_clear, c_tech, c_guide = st.columns([1, 1, 2.3, 1, 1, 1])

    with c_on:
        if st.button("⏻  開機", use_container_width=True, type="primary", key="btn_on"):
            ctrl = get_controller()
            ctrl.power_on(note="Dashboard 開機")
            st.session_state.is_running = True
            st.session_state.startup_begin = time.time()
            st.session_state.last_generated_ts = 0.0
            st.session_state.selected_fault = "NORMAL"
            st.session_state.applied_fault = "NORMAL"
            st.rerun()

    with c_off:
        if st.button("⏼  關機", use_container_width=True, type="secondary", key="btn_off"):
            ctrl = get_controller()
            ctrl.power_off(note="Dashboard 關機")
            st.session_state.is_running = False
            st.session_state.startup_begin = None
            st.session_state.last_generated_ts = 0.0
            st.session_state.applied_fault = "NORMAL"
            st.rerun()

    with c_mode:
        new_fault = st.selectbox(
            "模擬異常模式",
            options=FAULT_TYPES,
            index=FAULT_TYPES.index(st.session_state.selected_fault),
            format_func=lambda x: f"{x}　{FAULT_LABEL_MAP.get(x, '')}",
            key="fault_select",
        )
        if new_fault != st.session_state.selected_fault:
            st.session_state.selected_fault = new_fault
            st.rerun()

    with c_clear:
        ctrl = get_controller()
        is_off = ctrl.power_state == "OFF"
        if st.button(
            "🗑  清除資料",
            use_container_width=True,
            type="secondary",
            key="btn_clear",
            disabled=not is_off,
            help="請先關機再清除資料" if not is_off else "清除所有歷史資料並重建資料庫",
        ):
            from database import init_db
            db_path = "motor_monitor.db"
            if os.path.exists(db_path):
                os.remove(db_path)
            init_db(db_path)
            st.rerun()

    with c_tech:
        if st.button("📄 技術報告", use_container_width=True, key="btn_tech", type="secondary"):
            st.session_state.page = "tech"
            st.rerun()

    with c_guide:
        if st.button("📖 使用說明", use_container_width=True, key="btn_guide", type="secondary"):
            st.session_state.page = "guide"
            st.rerun()

    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(56,189,248,0.12);margin:10px 0 0 0;"/>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 完整資料表 HTML
# ---------------------------------------------------------------------------
def build_full_table_html(df: pd.DataFrame) -> str:
    display = df.sort_values("timestamp", ascending=False).head(100).copy()

    cols = [
        ("timestamp", "時間"),
        ("machine_state", "設備階段"),
        ("frequency_hz", "頻率 Hz"),
        ("current_a", "電流 A"),
        ("sync_rpm", "轉速 RPM"),
        ("slip_ratio", "轉差率"),
        ("torque_nm", "轉矩 N·m"),
        ("rule_fault_type", "Rule 判斷"),
        ("rule_level", "Rule 等級"),
        ("rule_confidence" if "rule_confidence" in df.columns else "rule_score", "Rule 信心"),
        ("ml_fault_type", "ML 判斷"),
        ("ml_level", "ML 等級"),
        ("ml_confidence", "ML 信心"),
        ("final_level", "綜合等級"),
    ]
    existing = [(c, l) for c, l in cols if c in display.columns]

    th = "".join(f"<th>{l}</th>" for _, l in existing)

    level_cls = {
        "NORMAL": "val-normal",
        "WARNING": "val-warn",
        "DANGER": "val-danger",
        "CRITICAL": "val-danger",
    }

    tbody = ""
    for _, row in display.iterrows():
        fl = str(row.get("final_level", "NORMAL"))
        row_style = ""
        if fl == "CRITICAL":
            row_style = 'style="background:rgba(248,113,113,0.06);"'
        elif fl in ("DANGER", "WARNING"):
            row_style = 'style="background:rgba(251,191,36,0.04);"'

        cells = ""
        for col, _ in existing:
            val = row[col]
            if col == "timestamp":
                val = val.strftime("%H:%M:%S") if hasattr(val, "strftime") else str(val)
                cells += f"<td>{val}</td>"
            elif col in ("frequency_hz", "current_a", "torque_nm"):
                cells += f"<td>{float(val):.2f}</td>"
            elif col == "sync_rpm":
                cells += f"<td>{int(float(val))}</td>"
            elif col == "slip_ratio":
                cells += f"<td>{float(val):.3f}</td>"
            elif col == "ml_confidence":
                cells += f"<td>{float(val):.0%}</td>"
            elif col in ("rule_confidence", "rule_score"):
                cells += f"<td>{int(val)}</td>"
            elif col in ("rule_fault_type", "ml_fault_type", "fault_type"):
                cells += f"<td>{fmt_fault(str(val))}</td>"
            elif col in ("rule_level", "ml_level", "final_level"):
                cls = level_cls.get(str(val), "val-muted")
                cells += f"<td class='{cls}'>{fmt_level(str(val))}</td>"
            else:
                cells += f"<td>{val}</td>"

        tbody += f"<tr {row_style}>{cells}</tr>"

    return f"""
<div class="sec-div">完整資料表（最近 100 筆）</div>
<div class="full-table-wrap">
  <table class="full-table">
    <thead><tr>{th}</tr></thead>
    <tbody>{tbody}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# 主 Dashboard HTML
# ---------------------------------------------------------------------------
def build_dashboard_html(
    df: pd.DataFrame,
    machine_state: str,
    applied_fault: str,
    startup_elapsed: float,
    is_running: bool,
    stats: dict,
    is_auto: bool = False,
    flow_items: list = None,
) -> str:
    flow_items = flow_items or []

    level_color = {
        "NORMAL": "var(--green)",
        "WARNING": "var(--amber)",
        "DANGER": "#fb923c",
        "CRITICAL": "var(--red)",
        "啟動中": "var(--text3)",
    }
    level_accent = {
        "NORMAL": "var(--green)",
        "WARNING": "var(--amber)",
        "DANGER": "#fb923c",
        "CRITICAL": "var(--red)",
        "啟動中": "var(--text3)",
    }

    if flow_items:
        boxes = []
        for i, (fault, level) in enumerate(flow_items):
            color = level_color.get(level, "var(--text3)")
            accent = level_accent.get(level, "var(--cyan)")
            label = fmt_level(level)
            value = fmt_fault(fault)
            is_latest = (i == len(flow_items) - 1)
            border = f"border:1px solid {accent};opacity:{1.0 if is_latest else 0.62};"
            boxes.append(
                f'<div class="flow-box" style="--accent:{accent};{border}">'
                f'<div class="flow-box-label">{label}</div>'
                f'<div class="flow-box-value" style="color:{color}">{value}</div>'
                f"</div>"
            )
        flow_html = '<div class="flow-track">'
        for i, box in enumerate(boxes):
            flow_html += box
            if i < len(boxes) - 1:
                flow_html += '<div class="flow-arrow">→</div>'
        flow_html += "</div>"
    else:
        flow_html = ""

    total = stats.get("total", 0)
    level_dist = stats.get("level_dist", {})
    warn_count = level_dist.get("WARNING", 0)
    danger_count = level_dist.get("DANGER", 0)
    critical_count = level_dist.get("CRITICAL", 0)
    alert_total = warn_count + danger_count + critical_count

    if not df.empty:
        latest = df.iloc[-1]
        l_freq = round(float(latest["frequency_hz"]), 1)
        l_curr = round(float(latest["current_a"]), 1)
        l_rpm = int(float(latest["sync_rpm"]))
        l_torq = round(float(latest["torque_nm"]), 1)
        l_mstate = str(latest["machine_state"])
        l_rfault = fmt_fault(str(latest["rule_fault_type"]))
        l_rlevel_raw = str(latest["rule_level"])
        l_rlevel = fmt_level(l_rlevel_raw)
        _score_col = "rule_confidence" if "rule_confidence" in latest.index else "rule_score"
        l_rscore = int(latest[_score_col])
        l_mlfault = fmt_fault(str(latest["ml_fault_type"]))
        l_mllevel = fmt_level(str(latest["ml_level"]))
        l_mlconf = round(float(latest["ml_confidence"]) * 100, 1)
        l_final = fmt_level(str(latest["final_level"]))
        l_final_raw = str(latest["final_level"])

        freq_pct = min(100, l_freq / 60 * 100)
        curr_pct = min(100, l_curr / 30 * 100)
        rpm_pct = min(100, l_rpm / 1800 * 100)
        torq_pct = min(100, l_torq / 120 * 100)

        final_cls = {
            "NORMAL": "val-normal",
            "WARNING": "val-warn",
            "DANGER": "val-danger",
            "CRITICAL": "val-danger",
        }.get(l_final_raw, "val-muted")

        rule_cls = {
            "NORMAL": "val-normal",
            "WARNING": "val-warn",
            "DANGER": "val-danger",
            "CRITICAL": "val-danger",
        }.get(l_rlevel_raw, "val-normal")
        ml_cls = "val-danger" if l_final_raw in ("DANGER", "CRITICAL") else "val-warn" if l_final_raw == "WARNING" else "val-normal"
    else:
        l_freq = l_curr = l_rpm = l_torq = 0
        l_rlevel_raw = "NORMAL"
        l_mstate = l_rfault = l_rlevel = l_mlfault = l_mllevel = l_final = "—"
        l_rscore = 0
        l_mlconf = 0.0
        freq_pct = curr_pct = rpm_pct = torq_pct = 0
        final_cls = rule_cls = ml_cls = "val-muted"
        l_final_raw = "NORMAL"

    if not df.empty:
        cdf = df.tail(MAX_CHART_POINTS)
        ts_labels = [t.strftime("%H:%M:%S") for t in cdf["timestamp"]]
        freq_ser = [round(float(v), 2) for v in cdf["frequency_hz"]]
        curr_ser = [round(float(v), 2) for v in cdf["current_a"]]
        rpm_ser = [int(float(v)) for v in cdf["sync_rpm"]]
        torq_ser = [round(float(v), 2) for v in cdf["torque_nm"]]
        _score_col2 = "rule_confidence" if "rule_confidence" in cdf.columns else "rule_score"
        rscore_ser = [int(v) for v in cdf[_score_col2]]
        mlconf_ser = [round(float(v) * 100, 1) for v in cdf["ml_confidence"]]
    else:
        ts_labels = freq_ser = curr_ser = rpm_ser = torq_ser = rscore_ser = mlconf_ser = []

    if not df.empty:
        adf = df[df["final_level"].isin(["WARNING", "DANGER", "CRITICAL"])].sort_values("timestamp", ascending=False).head(10)
        alert_rows = ""
        for _, r in adf.iterrows():
            fl = str(r["final_level"])
            sev_cls = {"WARNING": "sev-warn", "DANGER": "sev-danger", "CRITICAL": "sev-crit"}.get(fl, "sev-info")
            pill_cls = {"WARNING": "pill-warn", "DANGER": "pill-danger", "CRITICAL": "pill-crit"}.get(fl, "pill-ok")
            alert_rows += (
                f"<tr>"
                f"<td>{r['timestamp'].strftime('%H:%M:%S')}</td>"
                f"<td>{r['machine_state']}</td>"
                f"<td>{round(float(r['frequency_hz']),1)}</td>"
                f"<td>{round(float(r['current_a']),1)}</td>"
                f"<td>{fmt_fault(str(r['rule_fault_type']))}</td>"
                f"<td>{fmt_fault(str(r['ml_fault_type']))}</td>"
                f"<td>{round(float(r['ml_confidence'])*100,0):.0f}%</td>"
                f"<td class='{sev_cls}'><span class='pill {pill_cls}'>{fmt_level(fl)}</span></td>"
                f"</tr>"
            )
        alert_count_label = f"{len(adf)} 筆警報"
    else:
        alert_rows = '<tr><td colspan="8" style="text-align:center;color:var(--text3);padding:24px 0;">目前沒有異常事件</td></tr>'
        alert_count_label = "0 筆警報"

    full_table = build_full_table_html(df) if not df.empty else ""

    auto_section = ""
    if is_auto and not df.empty:
        adf = df[df["machine_state"] == "運轉中"].tail(10).copy()
        if not adf.empty:
            auto_rows = ""
            correct_count = 0
            total_count = 0
            for _, r in adf.iterrows():
                actual = str(r.get("rule_fault_type", "NORMAL"))
                ml_judge = str(r.get("ml_fault_type", "NORMAL"))
                ml_conf = round(float(r.get("ml_confidence", 0)) * 100, 0)

                is_correct = (actual == ml_judge)
                mark = "<span class='correct'>✅</span>" if is_correct else "<span class='wrong'>❌</span>"
                if is_correct:
                    correct_count += 1
                total_count += 1

                auto_rows += (
                    f"<tr>"
                    f"<td>{r['timestamp'].strftime('%H:%M:%S')}</td>"
                    f"<td>{fmt_fault(actual)}</td>"
                    f"<td>{fmt_fault(ml_judge)}</td>"
                    f"<td>{ml_conf:.0f}%</td>"
                    f"<td style='text-align:center'>{mark}</td>"
                    f"</tr>"
                )

            accuracy = round(correct_count / total_count * 100) if total_count > 0 else 0
            auto_section = f"""
<div class="sec-div">AUTO 模式即時驗證（最近 10 筆）</div>
<div class="auto-panel">
  <div class="auto-header">
    <div class="panel-title" style="color:var(--purple);">實際工況 vs ML 判斷</div>
    <span class="auto-badge">ML 準確率 {accuracy}%（{correct_count}/{total_count}）</span>
  </div>
  <table class="auto-table">
    <thead>
      <tr>
        <th>時間</th>
        <th>實際工況</th>
        <th>ML 判斷</th>
        <th>ML 信心</th>
        <th>正確</th>
      </tr>
    </thead>
    <tbody>{auto_rows}</tbody>
  </table>
</div>
"""

    live_dot = "#22d3a5" if is_running else "#4b6174"
    live_lbl = "ON" if is_running else "OFF"
    ms_color = {"關機": "#64748b", "啟動中": "#fbbf24", "運轉中": "#22d3a5"}.get(machine_state, "#64748b")
    ms_bg = {"關機": "rgba(100,116,139,.12)", "啟動中": "rgba(251,191,36,.12)", "運轉中": "rgba(34,211,165,.10)"}.get(machine_state, "rgba(100,116,139,.12)")
    ms_bd = {"關機": "rgba(100,116,139,.28)", "啟動中": "rgba(251,191,36,.34)", "運轉中": "rgba(34,211,165,.28)"}.get(machine_state, "rgba(100,116,139,.28)")
    fault_color = {
        "NORMAL": "#22d3a5",
        "OVERLOAD": "#fbbf24",
        "STALL": "#f87171",
        "LOAD_LOSS": "#fb923c",
        "BEARING_WEAR": "#fbbf24",
        "AUTO": "#a78bfa",
    }.get(applied_fault, "#22d3a5")
    fault_bg = {
        "NORMAL": "rgba(34,211,165,.10)",
        "OVERLOAD": "rgba(251,191,36,.12)",
        "STALL": "rgba(248,113,113,.14)",
        "LOAD_LOSS": "rgba(251,146,60,.12)",
        "BEARING_WEAR": "rgba(251,191,36,.12)",
        "AUTO": "rgba(167,139,250,.12)",
    }.get(applied_fault, "rgba(34,211,165,.10)")
    fault_bd = {
        "NORMAL": "rgba(34,211,165,.28)",
        "OVERLOAD": "rgba(251,191,36,.34)",
        "STALL": "rgba(248,113,113,.40)",
        "LOAD_LOSS": "rgba(251,146,60,.34)",
        "BEARING_WEAR": "rgba(251,191,36,.34)",
        "AUTO": "rgba(167,139,250,.30)",
    }.get(applied_fault, "rgba(34,211,165,.28)")

    ts_j = json.dumps(ts_labels)
    freq_j = json.dumps(freq_ser)
    curr_j = json.dumps(curr_ser)
    rpm_j = json.dumps(rpm_ser)
    torq_j = json.dumps(torq_ser)
    rs_j = json.dumps(rscore_ser)
    ml_j = json.dumps(mlconf_ser)

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.44.0/tabler-icons.min.css"/>
<style>
:root{{
  --bg:#0a0e14;
  --bg2:#0f1520;
  --bg3:#151d2e;
  --panel:#111827;
  --border:rgba(56,189,248,0.12);
  --border2:rgba(56,189,248,0.22);
  --cyan:#38bdf8;
  --cyan2:#0ea5e9;
  --green:#22d3a5;
  --amber:#fbbf24;
  --red:#f87171;
  --purple:#a78bfa;
  --orange:#fb923c;
  --text:#e2e8f0;
  --text2:#94a3b8;
  --text3:#64748b;
  --text4:#334155;
  --mono:'IBM Plex Mono',monospace;
  --sans:'IBM Plex Sans Condensed',sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{
  background:
    radial-gradient(circle at top right, rgba(56,189,248,.05), transparent 28%),
    radial-gradient(circle at top left, rgba(34,211,165,.03), transparent 24%),
    var(--bg);
  color:var(--text);
  font-family:var(--sans);
}}
.dash{{padding:16px 20px 44px;}}

.sec-div{{
  display:flex;align-items:center;gap:12px;margin:20px 0 12px;
  font-size:9px;font-family:var(--mono);color:var(--text3);
  text-transform:uppercase;letter-spacing:.12em;
}}
.sec-div::before,.sec-div::after{{
  content:'';flex:1;height:1px;background:linear-gradient(90deg, transparent, var(--border), transparent);
}}

.header{{
  position:relative;
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:18px;
  padding:18px 20px 16px;
  background:linear-gradient(180deg, rgba(17,24,39,.96), rgba(15,21,32,.96));
  border:1px solid var(--border);
  border-radius:16px;
  overflow:hidden;
  box-shadow:0 10px 26px rgba(0,0,0,.18);
}}
.header::before{{
  content:'';
  position:absolute;inset:0 0 auto 0;height:2px;
  background:linear-gradient(90deg,var(--cyan),var(--green),var(--purple));
  opacity:.84;
}}
.header::after{{
  content:'';
  position:absolute;right:-60px;top:-60px;width:180px;height:180px;
  background:radial-gradient(circle, rgba(56,189,248,.14), transparent 65%);
  pointer-events:none;
}}
.header-left{{display:flex;align-items:center;gap:14px;position:relative;z-index:1;}}
.logo-box{{
  width:44px;height:44px;border-radius:12px;
  background:linear-gradient(135deg,var(--cyan2),#0369a1);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  box-shadow:0 0 18px rgba(56,189,248,.16);
}}
.header-title{{font-size:18px;font-weight:700;letter-spacing:.02em;}}
.header-sub{{font-size:11px;color:var(--text2);font-family:var(--mono);margin-top:3px;letter-spacing:.05em;}}
.header-right{{display:flex;align-items:center;gap:12px;position:relative;z-index:1;}}
.header-badge-group{{display:flex;flex-direction:column;align-items:flex-end;gap:5px;}}
.hbadge-label{{font-size:9px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.08em;}}
.hbadge{{
  display:inline-block;padding:6px 14px;border-radius:999px;
  font-family:var(--mono);font-size:12px;font-weight:500;letter-spacing:.05em;border:1px solid;
  backdrop-filter: blur(2px);
}}
.live-pill{{display:flex;align-items:center;gap:8px;font-size:11px;font-family:var(--mono);color:var(--text2);padding-left:4px;}}
.live-dot{{width:8px;height:8px;border-radius:50%;background:{live_dot};box-shadow:0 0 8px {live_dot};animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:.35;}}}}

.flow-track{{display:flex;align-items:center;gap:0;margin-bottom:18px;}}
.flow-box{{
  flex:1;background:linear-gradient(180deg, rgba(17,24,39,.96), rgba(17,24,39,.92));
  border-radius:12px;padding:14px 16px;text-align:center;position:relative;overflow:hidden;
  box-shadow:0 6px 18px rgba(0,0,0,.14);
}}
.flow-box::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,var(--cyan));opacity:.72;
}}
.flow-box-label{{font-size:9px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;}}
.flow-box-value{{font-size:14px;font-family:var(--mono);font-weight:500;}}
.flow-arrow{{font-size:18px;color:var(--text3);padding:0 8px;flex-shrink:0;}}

.kpi-row{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;}}
.kpi-card{{
  background:linear-gradient(180deg, rgba(17,24,39,.96), rgba(17,24,39,.92));
  border:1px solid var(--border);border-radius:12px;padding:16px 16px;position:relative;overflow:hidden;
  box-shadow:0 6px 18px rgba(0,0,0,.14);
}}
.kpi-card::after{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,var(--cyan));opacity:.74;}}
.kpi-label{{font-size:10px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;}}
.kpi-value{{font-size:28px;font-family:var(--mono);font-weight:500;line-height:1;}}
.kpi-unit{{font-size:11px;color:var(--text3);margin-top:6px;font-family:var(--mono);}}

.main-grid{{display:grid;grid-template-columns:1fr 370px;gap:14px;}}
.sensor-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.sensor-card{{
  background:linear-gradient(180deg, rgba(17,24,39,.96), rgba(17,24,39,.92));
  border:1px solid var(--border);border-radius:12px;padding:15px 16px;
  box-shadow:0 6px 18px rgba(0,0,0,.14);
}}
.sensor-name{{font-size:10px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;display:flex;align-items:center;gap:6px;}}
.sensor-big{{font-size:34px;font-family:var(--mono);font-weight:500;line-height:1;}}
.sensor-unit{{font-size:12px;color:var(--text2);margin-left:4px;}}
.sensor-bar-wrap{{margin-top:12px;height:4px;background:rgba(255,255,255,.05);border-radius:999px;overflow:hidden;}}
.sensor-bar{{height:100%;border-radius:999px;transition:width .5s ease;}}

.status-panel{{
  background:linear-gradient(180deg, rgba(17,24,39,.96), rgba(17,24,39,.92));
  border:1px solid var(--border);border-radius:12px;padding:16px;
  display:flex;flex-direction:column;gap:14px;box-shadow:0 6px 18px rgba(0,0,0,.14);
}}
.panel-title{{font-size:10px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}}
.status-row{{display:flex;flex-direction:column;gap:8px;}}
.status-item-row{{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;background:var(--bg3);border-radius:8px;border:1px solid var(--border);}}
.status-key{{font-size:11px;color:var(--text2);font-family:var(--mono);}}
.status-val{{font-size:11px;font-family:var(--mono);font-weight:500;}}
.val-normal{{color:var(--green);}}
.val-warn{{color:var(--amber);}}
.val-danger{{color:var(--red);}}
.val-muted{{color:var(--text3);}}

.score-row{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;}}
.score-block{{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:12px 10px;text-align:center;}}
.score-num{{font-size:22px;font-family:var(--mono);font-weight:500;}}
.score-label{{font-size:9px;color:var(--text3);font-family:var(--mono);text-transform:uppercase;letter-spacing:.06em;margin-top:4px;line-height:1.45;}}

.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.chart-card{{
  background:linear-gradient(180deg, rgba(17,24,39,.96), rgba(17,24,39,.92));
  border:1px solid var(--border);border-radius:12px;padding:14px 16px;box-shadow:0 6px 18px rgba(0,0,0,.14);
}}
.chart-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}}
.chart-title{{font-size:11px;font-family:var(--mono);color:var(--text2);text-transform:uppercase;letter-spacing:.07em;}}
.badge-live{{font-size:9px;font-family:var(--mono);padding:3px 8px;border-radius:999px;background:rgba(34,211,165,.12);color:var(--green);border:1px solid rgba(34,211,165,.28);}}

.alert-section, .auto-panel{{
  background:linear-gradient(180deg, rgba(17,24,39,.96), rgba(17,24,39,.92));
  border:1px solid var(--border);border-radius:12px;padding:16px;box-shadow:0 6px 18px rgba(0,0,0,.14);
}}
.auto-panel{{border-color:rgba(167,139,250,0.28);}}
.auto-header,.alert-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}}
.auto-badge{{font-size:11px;font-family:var(--mono);padding:4px 10px;border-radius:999px;background:rgba(167,139,250,.12);border:1px solid rgba(167,139,250,.28);color:var(--purple);}}
.alert-count{{font-size:11px;font-family:var(--mono);padding:4px 10px;border-radius:999px;background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.28);color:var(--red);}}

.auto-table,.alert-table{{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono);}}
.auto-table th,.alert-table th{{
  text-align:left;padding:7px 10px;color:var(--text3);border-bottom:1px solid var(--border);
  font-size:10px;text-transform:uppercase;letter-spacing:.06em;background:rgba(255,255,255,.01);
}}
.auto-table td,.alert-table td{{
  padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.04);color:var(--text2);white-space:nowrap;
}}
.auto-table tr:hover td,.alert-table tr:hover td{{background:var(--bg3);}}
.correct{{color:var(--green);font-weight:500;}}
.wrong{{color:var(--red);font-weight:500;}}
.sev-warn{{color:var(--amber);}}
.sev-danger{{color:var(--orange);}}
.sev-crit{{color:var(--red);}}
.sev-info{{color:var(--cyan);}}
.pill{{padding:2px 8px;border-radius:999px;font-size:10px;}}
.pill-warn{{background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.28);color:var(--amber);}}
.pill-danger{{background:rgba(251,146,60,.12);border:1px solid rgba(251,146,60,.28);color:var(--orange);}}
.pill-crit{{background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.28);color:var(--red);}}
.pill-ok{{background:rgba(34,211,165,.10);border:1px solid rgba(34,211,165,.24);color:var(--green);}}

.full-table-wrap{{
  background:linear-gradient(180deg, rgba(17,24,39,.96), rgba(17,24,39,.92));
  border:1px solid var(--border);border-radius:12px;overflow-x:auto;overflow-y:auto;max-height:460px;
  box-shadow:0 6px 18px rgba(0,0,0,.14);
}}
.full-table{{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono);min-width:1100px;}}
.full-table thead{{position:sticky;top:0;z-index:2;}}
.full-table th{{
  text-align:left;padding:9px 12px;color:var(--text3);background:#0f1520;border-bottom:1px solid var(--border2);
  font-size:10px;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap;
}}
.full-table td{{padding:8px 12px;border-bottom:1px solid rgba(255,255,255,.04);color:var(--text2);white-space:nowrap;}}
.full-table tr:hover td{{background:rgba(56,189,248,0.04);}}

@media (max-width: 900px){{
  .header{{flex-direction:column;align-items:flex-start;gap:14px;}}
  .header-right{{flex-wrap:wrap;}}
  .kpi-row,.charts-row,.sensor-grid,.main-grid{{grid-template-columns:1fr;}}
}}
</style>
</head>
<body>
<div class="dash">

<div class="header">
  <div class="header-left">
    <div class="logo-box">
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
        <circle cx="11" cy="11" r="7" stroke="white" stroke-width="1.5" stroke-dasharray="3 2"/>
        <circle cx="11" cy="11" r="3" fill="white" opacity=".9"/>
        <line x1="11" y1="1" x2="11" y2="5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="11" y1="17" x2="11" y2="21" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="1" y1="11" x2="5" y2="11" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="17" y1="11" x2="21" y2="11" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="header-title">VFD MOTOR MONITORING SYSTEM</div>
      <div class="header-sub">變頻器馬達異常監測與風險分析 · v4.0 · Rule Confidence + Random Forest</div>
    </div>
  </div>
  <div class="header-right">
    <div class="header-badge-group">
      <div class="hbadge-label">設備階段</div>
      <div class="hbadge" style="background:{ms_bg};border-color:{ms_bd};color:{ms_color};">{machine_state}</div>
    </div>
    <div class="header-badge-group">
      <div class="hbadge-label">模擬模式</div>
      <div class="hbadge" style="background:{fault_bg};border-color:{fault_bd};color:{fault_color};">{fmt_fault(applied_fault)}</div>
    </div>
    <div class="header-badge-group">
      <div class="hbadge-label">已運行</div>
      <div class="hbadge" style="background:rgba(100,116,139,.12);border-color:rgba(100,116,139,.28);color:#94a3b8;">{round(startup_elapsed,1)}s</div>
    </div>
    <div class="live-pill">
      <div class="live-dot"></div>{live_lbl}
    </div>
  </div>
</div>

{flow_html}

<div class="sec-div">系統總覽</div>
<div class="kpi-row">
  <div class="kpi-card" style="--accent:var(--cyan)">
    <div class="kpi-label">總資料筆數</div>
    <div class="kpi-value">{total}</div>
    <div class="kpi-unit">records</div>
  </div>
  <div class="kpi-card" style="--accent:var(--amber)">
    <div class="kpi-label">警告</div>
    <div class="kpi-value" style="color:var(--amber);">{warn_count}</div>
    <div class="kpi-unit">WARNING</div>
  </div>
  <div class="kpi-card" style="--accent:#fb923c">
    <div class="kpi-label">危險</div>
    <div class="kpi-value" style="color:#fb923c;">{danger_count}</div>
    <div class="kpi-unit">DANGER</div>
  </div>
  <div class="kpi-card" style="--accent:var(--red)">
    <div class="kpi-label">緊急</div>
    <div class="kpi-value" style="color:var(--red);">{critical_count}</div>
    <div class="kpi-unit">CRITICAL</div>
  </div>
  <div class="kpi-card" style="--accent:var(--green)">
    <div class="kpi-label">累計告警</div>
    <div class="kpi-value">{alert_total}</div>
    <div class="kpi-unit">total alerts</div>
  </div>
</div>

<div class="sec-div">最新設備狀態</div>
<div class="main-grid">
  <div class="sensor-grid">
    <div class="sensor-card">
      <div class="sensor-name"><i class="ti ti-wave-sine"></i>頻率</div>
      <div><span class="sensor-big" style="color:var(--cyan)">{l_freq}</span><span class="sensor-unit">Hz</span></div>
      <div class="sensor-bar-wrap"><div class="sensor-bar" style="width:{freq_pct:.1f}%;background:var(--cyan)"></div></div>
    </div>
    <div class="sensor-card">
      <div class="sensor-name"><i class="ti ti-bolt"></i>電流</div>
      <div><span class="sensor-big" style="color:var(--amber)">{l_curr}</span><span class="sensor-unit">A</span></div>
      <div class="sensor-bar-wrap"><div class="sensor-bar" style="width:{curr_pct:.1f}%;background:var(--amber)"></div></div>
    </div>
    <div class="sensor-card">
      <div class="sensor-name"><i class="ti ti-rotate-clockwise"></i>轉速</div>
      <div><span class="sensor-big" style="color:var(--green)">{l_rpm}</span><span class="sensor-unit">RPM</span></div>
      <div class="sensor-bar-wrap"><div class="sensor-bar" style="width:{rpm_pct:.1f}%;background:var(--green)"></div></div>
    </div>
    <div class="sensor-card">
      <div class="sensor-name"><i class="ti ti-adjustments"></i>轉矩</div>
      <div><span class="sensor-big" style="color:var(--text)">{l_torq}</span><span class="sensor-unit">N·m</span></div>
      <div class="sensor-bar-wrap"><div class="sensor-bar" style="width:{torq_pct:.1f}%;background:#cbd5e1"></div></div>
    </div>
  </div>

  <div class="status-panel">
    <div>
      <div class="panel-title">診斷結果</div>
      <div class="status-row" style="margin-top:8px;">
        <div class="status-item-row"><span class="status-key">設備階段</span><span class="status-val val-muted">{l_mstate}</span></div>
        <div class="status-item-row"><span class="status-key">Rule 判斷</span><span class="status-val {rule_cls}">{l_rfault}</span></div>
        <div class="status-item-row"><span class="status-key">Rule 等級</span><span class="status-val {rule_cls}">{l_rlevel}</span></div>
        <div class="status-item-row"><span class="status-key">ML 判斷</span><span class="status-val {ml_cls}">{l_mlfault}</span></div>
        <div class="status-item-row"><span class="status-key">ML 等級</span><span class="status-val {ml_cls}">{l_mllevel}</span></div>
        <div class="status-item-row"><span class="status-key">綜合等級</span><span class="status-val {final_cls}">{l_final}</span></div>
      </div>
    </div>

    <div>
      <div class="panel-title">異常分數</div>
      <div class="score-row" style="margin-top:8px;">
        <div class="score-block">
          <div class="score-num {rule_cls}">{l_rscore}</div>
          <div class="score-label">Rule 信心<br/>Confidence</div>
        </div>
        <div class="score-block">
          <div class="score-num {ml_cls}">{l_mlconf}%</div>
          <div class="score-label">ML 信心<br/>Confidence</div>
        </div>
        <div class="score-block">
          <div class="score-num {final_cls}">{l_final}</div>
          <div class="score-label">綜合<br/>等級</div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="sec-div">感測器趨勢（最近 {MAX_CHART_POINTS} 筆）</div>
<div class="charts-row">
  <div class="chart-card">
    <div class="chart-header"><span class="chart-title">頻率 Hz / 轉速 RPM</span><span class="badge-live">LIVE</span></div>
    <div style="position:relative;height:180px;"><canvas id="c1"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-header"><span class="chart-title">電流 A</span><span class="badge-live">LIVE</span></div>
    <div style="position:relative;height:180px;"><canvas id="c2"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-header"><span class="chart-title">轉矩 N·m</span><span class="badge-live">LIVE</span></div>
    <div style="position:relative;height:180px;"><canvas id="c3"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-header"><span class="chart-title">Rule 信心（0~100）/ ML 信心（%）</span><span class="badge-live">LIVE</span></div>
    <div style="position:relative;height:180px;"><canvas id="c4"></canvas></div>
  </div>
</div>

{auto_section}

<div class="sec-div">最近異常事件</div>
<div class="alert-section">
  <div class="alert-header">
    <div class="panel-title">警報記錄</div>
    <span class="alert-count">{alert_count_label}</span>
  </div>
  <table class="alert-table">
    <thead>
      <tr>
        <th>時間</th><th>設備階段</th><th>頻率</th><th>電流</th>
        <th>Rule判斷</th><th>ML判斷</th><th>ML信心</th><th>綜合等級</th>
      </tr>
    </thead>
    <tbody>{alert_rows}</tbody>
  </table>
</div>

{full_table}

</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const L={ts_j},F={freq_j},C={curr_j},R={rpm_j},T={torq_j},RS={rs_j},ML={ml_j};
const base={{
  responsive:true,
  maintainAspectRatio:false,
  animation:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{
    x:{{display:false}},
    y:{{
      border:{{color:'rgba(255,255,255,0.06)'}},
      grid:{{color:'rgba(255,255,255,0.05)'}},
      ticks:{{font:{{family:'IBM Plex Mono',size:9}},color:'#64748b',maxTicksLimit:5}}
    }}
  }}
}};
function mkLine(id,datasets){{
  new Chart(document.getElementById(id),{{
    type:'line',
    data:{{
      labels:L,
      datasets:datasets.map(d=>({{
        label:d.l,
        data:d.d,
        borderColor:d.c,
        borderWidth:1.6,
        borderDash:d.dash||[],
        pointRadius:0,
        tension:0.32,
        fill:false
      }}))
    }},
    options:{{
      ...base,
      plugins:{{
        legend:{{
          display:datasets.length>1,
          labels:{{
            font:{{family:'IBM Plex Mono',size:10}},
            color:'#94a3b8',
            boxWidth:12,
            padding:10
          }}
        }}
      }}
    }}
  }});
}}
mkLine('c1',[{{l:'Hz',d:F,c:'#38bdf8'}},{{l:'RPM',d:R,c:'#22d3a5',dash:[3,3]}}]);
mkLine('c2',[{{l:'A',d:C,c:'#fbbf24'}}]);
mkLine('c3',[{{l:'N·m',d:T,c:'#cbd5e1'}}]);
mkLine('c4',[{{l:'Rule',d:RS,c:'#f87171'}},{{l:'ML',d:ML,c:'#a78bfa',dash:[4,2]}}]);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="VFD Motor Monitoring System",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_ml_model()
    init_session_state()

    if st.session_state.is_running:
        st_autorefresh(interval=DASHBOARD_REFRESH_INTERVAL_MS, key="motor_refresh")

    maybe_generate_data()
    render_top_controls()

    page = st.session_state.get("page", "dashboard")

    if page == "tech":
        if st.button("⬅ 返回監控面板", key="back_from_tech", type="secondary"):
            st.session_state.page = "dashboard"
            st.rerun()
        with open("static/technical_report.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=5000, scrolling=True)
        return

    if page == "guide":
        if st.button("⬅ 返回監控面板", key="back_from_guide", type="secondary"):
            st.session_state.page = "dashboard"
            st.rerun()
        with open("static/user_guide.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=4000, scrolling=True)
        return

    ctrl = get_controller()
    df = load_data()
    machine_state = ctrl.machine_state

    _sb = st.session_state.get("startup_begin", None)
    startup_elapsed = max(0.0, time.time() - _sb) if (_sb is not None and st.session_state.is_running) else 0.0
    if st.session_state.is_running and _sb is not None:
        ctrl._startup_begin = _sb

    flow_items = []
    if not df.empty:
        run_df = df[df["machine_state"] == "運轉中"].copy()
        if not run_df.empty:
            prev = None
            for _, r in run_df.iloc[::-1].iterrows():
                fault = str(r.get("rule_fault_type", "NORMAL"))
                level = str(r.get("final_level", "NORMAL"))
                if fault != prev:
                    flow_items.insert(0, (fault, level))
                    prev = fault
                if len(flow_items) >= 3:
                    break

    from VFD_simulator import get_current_fault
    applied_fault = get_current_fault(st.session_state.applied_fault)

    stats = fetch_stats()

    if df.empty and ctrl.power_state == "OFF":
        st.markdown(
            """
            <div style="
                text-align:center;
                padding:90px 0;
                font-family:'IBM Plex Mono',monospace;
                font-size:13px;
                color:#64748b;">
                按下「開機」開始模擬資料流
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    html = build_dashboard_html(
        df=df,
        machine_state=machine_state,
        applied_fault=applied_fault,
        startup_elapsed=startup_elapsed,
        is_running=st.session_state.is_running,
        stats=stats,
        is_auto=(st.session_state.selected_fault == "AUTO"),
        flow_items=flow_items,
    )

    components.html(html, height=3250, scrolling=False)


if __name__ == "__main__":
    main()
