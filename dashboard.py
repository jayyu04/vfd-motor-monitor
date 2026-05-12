from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

import os
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
DATA_GENERATE_INTERVAL_MS    = 300
DASHBOARD_REFRESH_INTERVAL_MS = 500
MAX_CHART_POINTS             = 200

FAULT_TYPES = ["NORMAL", "OVERLOAD", "STALL", "LOAD_LOSS", "BEARING_WEAR", "AUTO"]

FAULT_LABEL_MAP = {
    "NORMAL":       "正常",
    "OVERLOAD":     "過電流",
    "STALL":        "機械卡死",
    "LOAD_LOSS":    "負載斷裂",
    "BEARING_WEAR": "軸承磨損",
    "STARTUP":      "啟動中",
}

LEVEL_LABEL_MAP = {
    "NORMAL":   "正常",
    "WARNING":  "警告",
    "DANGER":   "危險",
    "CRITICAL": "緊急",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "is_running":         False,
        "selected_fault":     "NORMAL",
        "applied_fault":      "NORMAL",
        "last_generated_ts":  0.0,
        "page":               "dashboard",
        "startup_begin":      None,   # 開機時間，存在 session_state 避免 rerun 消失
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
    return get_controller().machine_state  # 關機/啟動中/運轉中


def maybe_generate_data() -> None:
    ctrl = get_controller()
    if not st.session_state.is_running or ctrl.power_state != "ON":
        return

    now = time.time()
    if now - st.session_state.last_generated_ts < DATA_GENERATE_INTERVAL_MS / 1000.0:
        return

    # 同步 startup_begin 到 control.py（Streamlit rerun 後單例可能重置）
    _sb = st.session_state.get("startup_begin", None)
    if _sb is not None:
        ctrl._startup_begin = _sb

    # 設定工況並產生資料
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
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
    [data-testid="stAppViewContainer"] { background: #0a0e14; }
    [data-testid="stHeader"]           { background: transparent; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"]  { display: none; }
    button[data-testid="baseButton-primary"] {
        background: rgba(34,211,165,0.08) !important;
        border: 1px solid rgba(34,211,165,0.45) !important;
        color: #22d3a5 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.06em;
        box-shadow: 0 0 8px rgba(34,211,165,0.2), inset 0 0 8px rgba(34,211,165,0.05) !important;
        text-shadow: 0 0 6px rgba(34,211,165,0.6) !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background: rgba(34,211,165,0.18) !important;
        border-color: rgba(34,211,165,0.8) !important;
        color: #ffffff !important;
        box-shadow: 0 0 18px rgba(34,211,165,0.5), inset 0 0 12px rgba(34,211,165,0.1) !important;
        text-shadow: 0 0 10px rgba(34,211,165,1.0) !important;
    }
    button[data-testid="baseButton-secondary"] {
        background: rgba(248,113,113,0.08) !important;
        border: 1px solid rgba(248,113,113,0.4) !important;
        color: #f87171 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.06em;
        box-shadow: 0 0 8px rgba(248,113,113,0.15), inset 0 0 8px rgba(248,113,113,0.05) !important;
        text-shadow: 0 0 6px rgba(248,113,113,0.6) !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background: rgba(248,113,113,0.18) !important;
        border-color: rgba(248,113,113,0.8) !important;
        color: #ffffff !important;
        box-shadow: 0 0 18px rgba(248,113,113,0.5), inset 0 0 12px rgba(248,113,113,0.1) !important;
        text-shadow: 0 0 10px rgba(248,113,113,1.0) !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: #111827 !important;
        border: 1px solid rgba(56,189,248,0.22) !important;
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
    }
    div[data-testid="stSelectbox"] label {
        color: #4b6174 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    iframe { border: none !important; }


    </style>
    """, unsafe_allow_html=True)

    ms      = get_current_machine_state()
    elapsed = get_startup_elapsed_sec()
    applied = st.session_state.applied_fault

    def badge(label, value, bg, bd, tc):
        return (
            f"<div style=\"font-size:9px;color:#4b6174;font-family:'IBM Plex Mono',monospace;"
            f"text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;\">{label}</div>"
            f"<span style=\"display:inline-block;padding:5px 14px;border-radius:4px;"
            f"font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:500;"
            f"letter-spacing:.05em;background:{bg};border:1px solid {bd};color:{tc};\">{value}</span>"
        )

    ms_cfg = {
        "OFF":     ("rgba(75,97,116,.2)",    "rgba(75,97,116,.4)",    "#4b6174"),
        "STARTUP": ("rgba(251,191,36,.15)",  "rgba(251,191,36,.4)",   "#fbbf24"),
        "RUNNING": ("rgba(34,211,165,.12)",  "rgba(34,211,165,.35)",  "#22d3a5"),
    }
    mbg, mbd, mtc = ms_cfg.get(ms, ms_cfg["OFF"])

    fault_cfg = {
        "NORMAL":       ("#22d3a5", "rgba(34,211,165,.12)",  "rgba(34,211,165,.35)"),
        "OVERLOAD":     ("#fbbf24", "rgba(251,191,36,.15)",  "rgba(251,191,36,.4)"),
        "STALL":        ("#f87171", "rgba(248,113,113,.18)", "rgba(248,113,113,.5)"),
        "LOAD_LOSS":    ("#f87171", "rgba(248,113,113,.12)", "rgba(248,113,113,.3)"),
        "BEARING_WEAR": ("#fbbf24", "rgba(251,191,36,.15)",  "rgba(251,191,36,.4)"),
    }
    amc, ambg, ambd = fault_cfg.get(applied, fault_cfg["NORMAL"])

    dot_color  = "#22d3a5" if st.session_state.is_running else "#4b6174"
    live_label = "LIVE" if st.session_state.is_running else "IDLE"
    live_html = (
        f"<div style='display:flex;flex-direction:column;align-items:center;gap:5px;padding-top:2px;'>"
        f"<div style='width:9px;height:9px;border-radius:50%;background:{dot_color};"
        f"box-shadow:0 0 7px {dot_color};'></div>"
        f"<span style=\"font-family:'IBM Plex Mono',monospace;font-size:9px;color:#4b6174;"
        f"text-transform:uppercase;letter-spacing:.08em;\">{live_label}</span>"
        f"</div>"
    )

    c_on, c_off, c_mode, c_clear, c_tech, c_guide = st.columns([1, 1, 2, 1, 1, 1])

    with c_on:
        if st.button("⏻  開機", use_container_width=True, type="primary", key="btn_on"):
            ctrl = get_controller()
            ctrl.power_on(note="Dashboard 開機")
            st.session_state.is_running        = True
            st.session_state.startup_begin     = time.time()
            st.session_state.last_generated_ts = 0.0
            st.session_state.selected_fault    = "NORMAL"
            st.session_state.applied_fault     = "NORMAL"
            st.rerun()

    with c_off:
        if st.button("⏼  關機", use_container_width=True, type="secondary", key="btn_off"):
            ctrl = get_controller()
            ctrl.power_off(note="Dashboard 關機")
            st.session_state.is_running        = False
            st.session_state.startup_begin     = None
            st.session_state.last_generated_ts = 0.0
            st.session_state.applied_fault     = "NORMAL"
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


    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(56,189,248,0.12);margin:8px 0 0 0;"/>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 完整資料表 HTML
# ---------------------------------------------------------------------------
def build_full_table_html(df: pd.DataFrame) -> str:
    display = df.sort_values("timestamp", ascending=False).head(100).copy()

    cols = [
        ("timestamp",       "時間"),
        ("machine_state",   "設備階段"),
        ("frequency_hz",    "頻率 Hz"),
        ("current_a",       "電流 A"),
        ("sync_rpm",        "轉速 RPM"),
        ("slip_ratio",      "轉差率"),
        ("torque_nm",       "轉矩 N·m"),
        ("rule_fault_type", "Rule 判斷"),
        ("rule_level",      "Rule 等級"),
        ("rule_confidence" if "rule_confidence" in df.columns else "rule_score", "Rule 信心"),
        ("ml_fault_type",   "ML 判斷"),
        ("ml_level",        "ML 等級"),
        ("ml_confidence",   "ML 信心"),
        ("final_level",     "綜合等級"),
    ]
    existing = [(c, l) for c, l in cols if c in display.columns]

    th = "".join(f"<th>{l}</th>" for _, l in existing)

    level_cls = {
        "NORMAL": "val-normal", "WARNING": "val-warn",
        "DANGER": "val-danger", "CRITICAL": "val-danger",
    }

    tbody = ""
    for _, row in display.iterrows():
        fl = str(row.get("final_level", "NORMAL"))
        row_style = ""
        if fl == "CRITICAL":
            row_style = 'style="background:rgba(248,113,113,0.07);"'
        elif fl in ("DANGER", "WARNING"):
            row_style = 'style="background:rgba(251,191,36,0.05);"'

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
            elif col == "rule_confidence":
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

    # ── 工況流程 ──
    flow_items = flow_items or []
    level_color = {
        "NORMAL":   "var(--green)",
        "WARNING":  "var(--amber)",
        "DANGER":   "#fb923c",
        "CRITICAL": "var(--red)",
        "啟動中":    "var(--text3)",
    }
    level_accent = {
        "NORMAL":   "var(--green)",
        "WARNING":  "var(--amber)",
        "DANGER":   "#fb923c",
        "CRITICAL": "var(--red)",
        "啟動中":    "var(--text3)",
    }

    if flow_items:
        boxes = []
        for i, (fault, level) in enumerate(flow_items):
            color  = level_color.get(level, "var(--text3)")
            accent = level_accent.get(level, "var(--cyan)")
            label  = fmt_level(level)
            value  = fmt_fault(fault)
            is_latest = (i == len(flow_items) - 1)
            border = f"border:1px solid {accent};opacity:{1.0 if is_latest else 0.6};"
            boxes.append(
                f'<div class="flow-box" style="--accent:{accent};{border}">'
                f'<div class="flow-box-label">{label}</div>'
                f'<div class="flow-box-value" style="color:{color}">{value}</div>'
                f'</div>'
            )
        flow_html = '<div class="flow-track">'
        for i, box in enumerate(boxes):
            flow_html += box
            if i < len(boxes) - 1:
                flow_html += '<div class="flow-arrow">→</div>'
        flow_html += '</div>'
    else:
        flow_html = ""

    # ── KPI ──
    total       = stats.get("total", 0)
    level_dist  = stats.get("level_dist", {})
    fault_dist  = stats.get("fault_dist", {})
    warn_count  = level_dist.get("WARNING", 0)
    danger_count = level_dist.get("DANGER", 0)
    critical_count = level_dist.get("CRITICAL", 0)
    alert_total = warn_count + danger_count + critical_count

    # ── Latest ──
    if not df.empty:
        latest     = df.iloc[-1]
        l_freq     = round(float(latest["frequency_hz"]), 1)
        l_curr     = round(float(latest["current_a"]), 1)
        l_rpm      = int(float(latest["sync_rpm"]))
        l_torq     = round(float(latest["torque_nm"]), 1)
        l_mstate   = str(latest["machine_state"])
        l_rfault   = fmt_fault(str(latest["rule_fault_type"]))
        l_rlevel   = fmt_level(str(latest["rule_level"]))
        # 相容舊版資料庫（rule_score）和新版（rule_confidence）
        _score_col = "rule_confidence" if "rule_confidence" in latest.index else "rule_score"
        l_rscore   = int(latest[_score_col])
        l_mlfault  = fmt_fault(str(latest["ml_fault_type"]))
        l_mllevel  = fmt_level(str(latest["ml_level"]))
        l_mlconf   = round(float(latest["ml_confidence"]) * 100, 1)
        l_final    = fmt_level(str(latest["final_level"]))
        l_final_raw = str(latest["final_level"])

        freq_pct = min(100, l_freq / 60 * 100)
        curr_pct = min(100, l_curr / 30 * 100)
        rpm_pct  = min(100, l_rpm / 1800 * 100)
        torq_pct = min(100, l_torq / 120 * 100)

        final_cls = {
            "NORMAL": "val-normal", "WARNING": "val-warn",
            "DANGER": "val-danger", "CRITICAL": "val-danger",
        }.get(l_final_raw, "val-muted")

        rule_cls = "val-danger" if l_rscore >= 75 else "val-warn" if l_rscore >= 25 else "val-normal"
        ml_cls   = "val-danger" if l_final_raw in ("DANGER", "CRITICAL") else "val-warn" if l_final_raw == "WARNING" else "val-normal"
    else:
        l_freq = l_curr = l_rpm = l_torq = 0
        l_mstate = l_rfault = l_rlevel = l_mlfault = l_mllevel = l_final = "—"
        l_rscore = 0
        l_mlconf = 0.0
        freq_pct = curr_pct = rpm_pct = torq_pct = 0
        final_cls = rule_cls = ml_cls = "val-muted"
        l_final_raw = "NORMAL"

    # ── Chart series ──
    if not df.empty:
        cdf        = df.tail(MAX_CHART_POINTS)
        ts_labels  = [t.strftime("%H:%M:%S") for t in cdf["timestamp"]]
        freq_ser   = [round(float(v), 2) for v in cdf["frequency_hz"]]
        curr_ser   = [round(float(v), 2) for v in cdf["current_a"]]
        rpm_ser    = [int(float(v)) for v in cdf["sync_rpm"]]
        torq_ser   = [round(float(v), 2) for v in cdf["torque_nm"]]
        _score_col2 = "rule_confidence" if "rule_confidence" in cdf.columns else "rule_score"
        rscore_ser = [int(v) for v in cdf[_score_col2]]
        mlconf_ser = [round(float(v) * 100, 1) for v in cdf["ml_confidence"]]
    else:
        ts_labels = freq_ser = curr_ser = rpm_ser = torq_ser = rscore_ser = mlconf_ser = []

    # ── Alert rows ──
    if not df.empty:
        adf = df[df["final_level"].isin(["WARNING", "DANGER", "CRITICAL"])] \
                .sort_values("timestamp", ascending=False).head(10)
        alert_rows = ""
        for _, r in adf.iterrows():
            fl  = str(r["final_level"])
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

    # AUTO 模式對比區塊
    auto_section = ""
    if is_auto and not df.empty:
        adf = df[df["machine_state"] == "運轉中"].tail(10).copy()
        if not adf.empty:
            auto_rows = ""
            correct_count = 0
            total_count = 0
            for _, r in adf.iterrows():
                # 實際工況：從 rule_fault_type 讀（VFD_simulator 決定的）
                actual   = str(r.get("rule_fault_type", "NORMAL"))
                ml_judge = str(r.get("ml_fault_type",   "NORMAL"))
                ml_conf  = round(float(r.get("ml_confidence", 0)) * 100, 0)

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

    # ── Header colors ──
    live_dot  = "#22d3a5" if is_running else "#4b6174"
    live_lbl  = "ON" if is_running else "OFF"
    ms_color  = {"關機": "#4b6174", "啟動中": "#fbbf24", "運轉中": "#22d3a5"}.get(machine_state, "#4b6174")
    ms_bg     = {"關機": "rgba(75,97,116,.15)", "啟動中": "rgba(251,191,36,.15)", "運轉中": "rgba(34,211,165,.12)"}.get(machine_state, "rgba(75,97,116,.15)")
    ms_bd     = {"關機": "rgba(75,97,116,.35)", "啟動中": "rgba(251,191,36,.4)",  "運轉中": "rgba(34,211,165,.35)"}.get(machine_state, "rgba(75,97,116,.35)")
    fault_color = {
        "NORMAL": "#22d3a5", "OVERLOAD": "#fbbf24",
        "STALL": "#f87171", "LOAD_LOSS": "#f87171", "BEARING_WEAR": "#fbbf24",
        "AUTO": "#a78bfa",
    }.get(applied_fault, "#22d3a5")
    fault_bg  = {
        "NORMAL": "rgba(34,211,165,.12)", "OVERLOAD": "rgba(251,191,36,.15)",
        "STALL": "rgba(248,113,113,.18)", "LOAD_LOSS": "rgba(248,113,113,.12)",
        "BEARING_WEAR": "rgba(251,191,36,.15)", "AUTO": "rgba(167,139,250,.12)",
    }.get(applied_fault, "rgba(34,211,165,.12)")
    fault_bd  = {
        "NORMAL": "rgba(34,211,165,.35)", "OVERLOAD": "rgba(251,191,36,.4)",
        "STALL": "rgba(248,113,113,.5)",  "LOAD_LOSS": "rgba(248,113,113,.3)",
        "BEARING_WEAR": "rgba(251,191,36,.4)", "AUTO": "rgba(167,139,250,.35)",
    }.get(applied_fault, "rgba(34,211,165,.35)")

    ts_j   = json.dumps(ts_labels)
    freq_j = json.dumps(freq_ser)
    curr_j = json.dumps(curr_ser)
    rpm_j  = json.dumps(rpm_ser)
    torq_j = json.dumps(torq_ser)
    rs_j   = json.dumps(rscore_ser)
    ml_j   = json.dumps(mlconf_ser)

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;500;600&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.44.0/tabler-icons.min.css"/>
<style>
:root{{
  --bg:#0a0e14;--bg3:#151d2e;--panel:#111827;
  --border:rgba(56,189,248,0.12);--border2:rgba(56,189,248,0.22);
  --cyan:#38bdf8;--cyan2:#0ea5e9;--green:#22d3a5;--amber:#fbbf24;
  --red:#f87171;--purple:#a78bfa;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#4b6174;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans Condensed',sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{background:var(--bg);color:var(--text);font-family:var(--sans);}}
.dash{{padding:14px 20px 40px;}}
.sec-div{{display:flex;align-items:center;gap:10px;margin:18px 0 12px;
  font-size:9px;font-family:var(--mono);color:var(--text3);
  text-transform:uppercase;letter-spacing:.1em;}}
.sec-div::before,.sec-div::after{{content:'';flex:1;height:1px;background:var(--border);}}
.header{{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:18px;padding-bottom:16px;border-bottom:1px solid var(--border);}}
.header-left{{display:flex;align-items:center;gap:14px;}}
.logo-box{{width:42px;height:42px;background:linear-gradient(135deg,var(--cyan2),#0369a1);
  border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}}
.header-title{{font-size:18px;font-weight:600;letter-spacing:.02em;}}
.header-sub{{font-size:11px;color:var(--text2);font-family:var(--mono);margin-top:3px;letter-spacing:.05em;}}
.header-right{{display:flex;align-items:center;gap:20px;}}
.live-pill{{display:flex;align-items:center;gap:7px;font-size:11px;font-family:var(--mono);color:var(--text2);}}
.header-doc-links{{display:flex;gap:8px;margin-left:4px;}}
.doc-btn{{font-size:10px;font-family:var(--mono);padding:5px 12px;border-radius:4px;
  text-decoration:none;letter-spacing:.04em;transition:opacity .2s;}}
.doc-btn:hover{{opacity:.8;}}
.doc-btn-cyan{{background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.25);color:var(--cyan);}}
.doc-btn-green{{background:rgba(34,211,165,.1);border:1px solid rgba(34,211,165,.25);color:var(--green);}}
.live-dot{{width:8px;height:8px;border-radius:50%;background:{live_dot};
  box-shadow:0 0 7px {live_dot};animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:.35;}}}}
.header-badge-group{{display:flex;flex-direction:column;align-items:flex-end;gap:5px;}}
.hbadge-label{{font-size:9px;font-family:var(--mono);color:var(--text3);text-transform:uppercase;letter-spacing:.08em;}}
.hbadge{{display:inline-block;padding:5px 14px;border-radius:4px;
  font-family:var(--mono);font-size:12px;font-weight:500;
  letter-spacing:.05em;border:1px solid;}}
.flow-track{{display:flex;align-items:center;gap:0;margin-bottom:18px;}}
.flow-box{{flex:1;background:var(--panel);border:1px solid var(--border);
  border-radius:8px;padding:14px 16px;text-align:center;position:relative;overflow:hidden;}}
.flow-box::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,var(--cyan));opacity:.7;}}
.flow-box-label{{font-size:9px;font-family:var(--mono);color:var(--text3);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;}}
.flow-box-value{{font-size:14px;font-family:var(--mono);font-weight:500;}}
.flow-arrow{{font-size:18px;color:var(--text3);padding:0 8px;flex-shrink:0;}}
.kpi-row{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;}}
.kpi-card{{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:14px 16px;position:relative;overflow:hidden;}}
.kpi-card::after{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,var(--cyan));opacity:.7;}}
.kpi-label{{font-size:10px;font-family:var(--mono);color:var(--text3);
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;}}
.kpi-value{{font-size:26px;font-family:var(--mono);font-weight:500;line-height:1;}}
.kpi-unit{{font-size:11px;color:var(--text3);margin-top:4px;font-family:var(--mono);}}
.main-grid{{display:grid;grid-template-columns:1fr 360px;gap:14px;}}
.sensor-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.sensor-card{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 16px;}}
.sensor-name{{font-size:10px;font-family:var(--mono);color:var(--text3);
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;
  display:flex;align-items:center;gap:6px;}}
.sensor-big{{font-size:32px;font-family:var(--mono);font-weight:500;line-height:1;}}
.sensor-unit{{font-size:12px;color:var(--text2);margin-left:4px;}}
.sensor-bar-wrap{{margin-top:10px;height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;}}
.sensor-bar{{height:100%;border-radius:2px;transition:width .5s ease;}}
.status-panel{{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:16px;display:flex;flex-direction:column;gap:14px;}}
.panel-title{{font-size:10px;font-family:var(--mono);color:var(--text3);
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;}}
.status-row{{display:flex;flex-direction:column;gap:7px;}}
.status-item-row{{display:flex;align-items:center;justify-content:space-between;
  padding:7px 10px;background:var(--bg3);border-radius:6px;border:1px solid var(--border);}}
.status-key{{font-size:11px;color:var(--text2);font-family:var(--mono);}}
.status-val{{font-size:11px;font-family:var(--mono);font-weight:500;}}
.val-normal{{color:var(--green);}} .val-warn{{color:var(--amber);}}
.val-danger{{color:var(--red);}}   .val-muted{{color:var(--text3);}}
.score-row{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;}}
.score-block{{background:var(--bg3);border:1px solid var(--border);
  border-radius:6px;padding:10px;text-align:center;}}
.score-num{{font-size:22px;font-family:var(--mono);font-weight:500;}}
.score-label{{font-size:9px;color:var(--text3);font-family:var(--mono);
  text-transform:uppercase;letter-spacing:.06em;margin-top:3px;}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.chart-card{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 16px;}}
.chart-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}}
.chart-title{{font-size:11px;font-family:var(--mono);color:var(--text2);
  text-transform:uppercase;letter-spacing:.07em;}}
.badge-live{{font-size:9px;font-family:var(--mono);padding:2px 7px;border-radius:3px;
  background:rgba(34,211,165,.15);color:var(--green);border:1px solid rgba(34,211,165,.3);}}
.alert-section{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:16px;}}
.auto-panel{{background:var(--panel);border:1px solid rgba(167,139,250,0.3);border-radius:8px;padding:16px;}}
.auto-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}}
.auto-badge{{font-size:11px;font-family:var(--mono);padding:3px 9px;border-radius:4px;
  background:rgba(167,139,250,.15);border:1px solid rgba(167,139,250,.3);color:var(--purple);}}
.auto-table{{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono);}}
.auto-table th{{text-align:left;padding:6px 10px;color:var(--text3);
  border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase;letter-spacing:.06em;}}
.auto-table td{{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.04);color:var(--text2);white-space:nowrap;}}
.auto-table tr:hover td{{background:var(--bg3);}}
.correct{{color:var(--green);font-weight:500;}}
.wrong{{color:var(--red);font-weight:500;}}
.alert-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}}
.alert-count{{font-size:11px;font-family:var(--mono);padding:3px 9px;border-radius:4px;
  background:rgba(248,113,113,.15);border:1px solid rgba(248,113,113,.3);color:var(--red);}}
.alert-table{{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono);}}
.alert-table th{{text-align:left;padding:6px 10px;color:var(--text3);
  border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase;letter-spacing:.06em;}}
.alert-table td{{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.04);
  color:var(--text2);white-space:nowrap;}}
.alert-table tr:hover td{{background:var(--bg3);}}
.sev-warn{{color:var(--amber);}} .sev-danger{{color:var(--orange);}} .sev-crit{{color:var(--red);}} .sev-info{{color:var(--cyan);}}
.pill{{padding:2px 7px;border-radius:3px;font-size:10px;}}
.pill-warn{{background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.3);color:var(--amber);}}
.pill-danger{{background:rgba(251,146,60,.12);border:1px solid rgba(251,146,60,.3);color:var(--orange);}}
.pill-crit{{background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.3);color:var(--red);}}
.pill-ok{{background:rgba(34,211,165,.1);border:1px solid rgba(34,211,165,.3);color:var(--green);}}
.full-table-wrap{{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  overflow-x:auto;overflow-y:auto;max-height:460px;}}
.full-table{{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono);min-width:1100px;}}
.full-table thead{{position:sticky;top:0;z-index:2;}}
.full-table th{{text-align:left;padding:8px 12px;color:var(--text3);
  background:#0f1520;border-bottom:1px solid var(--border2);
  font-size:10px;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap;}}
.full-table td{{padding:7px 12px;border-bottom:1px solid rgba(255,255,255,.04);
  color:var(--text2);white-space:nowrap;}}
.full-table tr:hover td{{background:rgba(56,189,248,0.04);}}
</style>
</head>
<body>
<div class="dash">

<!-- ══ Header ══ -->
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
      <div class="header-sub">變頻器馬達異常監測與風險分析 · v3.0 · Rule-based Anomaly Score + Random Forest</div>
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
      <div class="hbadge" style="background:rgba(75,97,116,.15);border-color:rgba(75,97,116,.35);color:#94a3b8;">{round(startup_elapsed,1)}s</div>
    </div>
    <div class="live-pill">
      <div class="live-dot"></div>{live_lbl}
    </div>
  </div>
</div>

{flow_html}
<!-- ══ KPI ══ -->
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
  <div class="kpi-card" style="--accent:var(--red)">
    <div class="kpi-label">危險</div>
    <div class="kpi-value" style="color:var(--red);">{danger_count}</div>
    <div class="kpi-unit">DANGER</div>
  </div>
  <div class="kpi-card" style="--accent:var(--purple)">
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

<!-- ══ Sensors + Status ══ -->
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
      <div class="sensor-bar-wrap"><div class="sensor-bar" style="width:{torq_pct:.1f}%;background:var(--text2)"></div></div>
    </div>
  </div>

  <div class="status-panel">
    <div>
      <div class="panel-title">診斷結果</div>
      <div class="status-row" style="margin-top:8px;">
        <div class="status-item-row">
          <span class="status-key">設備階段</span>
          <span class="status-val val-muted">{l_mstate}</span>
        </div>
        <div class="status-item-row">
          <span class="status-key">Rule 判斷</span>
          <span class="status-val {rule_cls}">{l_rfault}</span>
        </div>
        <div class="status-item-row">
          <span class="status-key">Rule 等級</span>
          <span class="status-val {rule_cls}">{l_rlevel}</span>
        </div>
        <div class="status-item-row">
          <span class="status-key">ML 判斷</span>
          <span class="status-val {ml_cls}">{l_mlfault}</span>
        </div>
        <div class="status-item-row">
          <span class="status-key">ML 等級</span>
          <span class="status-val {ml_cls}">{l_mllevel}</span>
        </div>
        <div class="status-item-row">
          <span class="status-key">綜合等級</span>
          <span class="status-val {final_cls}">{l_final}</span>
        </div>
      </div>
    </div>
    <div>
      <div class="panel-title">異常分數</div>
      <div class="score-row" style="margin-top:8px;">
        <div class="score-block">
          <div class="score-num {rule_cls}">{l_rscore}</div>
          <div class="score-label">Rule Score<br/>0~100</div>
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

<!-- ══ Charts ══ -->
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
    <div class="chart-header"><span class="chart-title">Rule 分數（0~100）/ ML 信心（%）</span><span class="badge-live">LIVE</span></div>
    <div style="position:relative;height:180px;"><canvas id="c4"></canvas></div>
  </div>
</div>

{auto_section}

<!-- ══ Alert table ══ -->
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
  responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{
    x:{{display:false}},
    y:{{border:{{color:'rgba(255,255,255,0.06)'}},grid:{{color:'rgba(255,255,255,0.05)'}},
       ticks:{{font:{{family:'IBM Plex Mono',size:9}},color:'#4b6174',maxTicksLimit:5}}}}
  }}
}};
function mkLine(id,datasets){{
  new Chart(document.getElementById(id),{{
    type:'line',
    data:{{labels:L,datasets:datasets.map(d=>{{return{{
      label:d.l,data:d.d,borderColor:d.c,borderWidth:1.5,
      borderDash:d.dash||[],pointRadius:0,tension:0.3,fill:false
    }}}})  }},
    options:{{...base,plugins:{{legend:{{
      display:datasets.length>1,
      labels:{{font:{{family:'IBM Plex Mono',size:10}},color:'#94a3b8',boxWidth:12,padding:10}}
    }}}}}}
  }});
}}
mkLine('c1',[{{l:'Hz',d:F,c:'#38bdf8'}},{{l:'RPM',d:R,c:'#22d3a5',dash:[3,3]}}]);
mkLine('c2',[{{l:'A',d:C,c:'#fbbf24'}}]);
mkLine('c3',[{{l:'N·m',d:T,c:'#cbd5e1'}}]);
mkLine('c4',[{{l:'Rule Score',d:RS,c:'#f87171'}},{{l:'ML Conf%',d:ML,c:'#a78bfa',dash:[4,2]}}]);
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

    init_ml_model()      # 確保 ML 只初始化一次，Streamlit rerun 不重新訓練
    init_session_state()

    # autorefresh 要在最前面，確保持續觸發
    if st.session_state.is_running:
        st_autorefresh(interval=DASHBOARD_REFRESH_INTERVAL_MS, key="motor_refresh")

    maybe_generate_data()
    render_top_controls()

    # ── 頁面切換 ──
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


    ctrl          = get_controller()
    df            = load_data()
    machine_state = ctrl.machine_state
    # 從 session_state 計算 elapsed_sec，避免 Streamlit rerun 時單例消失
    _sb = st.session_state.get("startup_begin", None)
    startup_elapsed = max(0.0, time.time() - _sb) if (_sb is not None and st.session_state.is_running) else 0.0
    # 同步回 control.py（讓 main.tick 拿到正確的 elapsed_sec）
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

    # 從 VFD_simulator 直接讀取目前實際工況（AUTO 模式會回傳實際切換到的工況）
    from VFD_simulator import get_current_fault
    applied_fault = get_current_fault(st.session_state.applied_fault)

    stats = fetch_stats()

    if df.empty and ctrl.power_state == "OFF":
        st.markdown(
            '<div style="text-align:center;padding:80px 0;'
            "font-family:'IBM Plex Mono',monospace;font-size:13px;color:#4b6174;\">"
            "按下「開機」開始模擬資料流</div>",
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

    components.html(html, height=3200, scrolling=False)


if __name__ == "__main__":
    main()
