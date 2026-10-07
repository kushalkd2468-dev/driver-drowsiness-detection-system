"""
Driver Drowsiness — Streamlit Analytics Dashboard
Run: streamlit run dashboard.py
Reads: session_log.csv, driver_profiles.json, reports/*.json
Auto-refreshes every 3 s while a live session is running.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import time
import glob
import datetime
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DrowseSense — Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (industrial/utilitarian dark theme) ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@700;900&display=swap');

/* Root palette */
:root {
    --bg:        #090c10;
    --panel:     #0f1318;
    --border:    #1e2530;
    --accent:    #00e5ff;
    --warn:      #ff8c00;
    --danger:    #ff2d2d;
    --ok:        #00e676;
    --text:      #c8d0dc;
    --subtext:   #5a6478;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow', sans-serif;
    --cond:      'Barlow Condensed', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
div[data-testid="metric-container"] label {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    color: var(--subtext) !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--cond) !important;
    font-size: 2.1rem !important;
    font-weight: 900 !important;
    color: var(--accent) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
}

/* Section headers */
h1, h2, h3 {
    font-family: var(--cond) !important;
    font-weight: 900 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
}
h1 { font-size: 2.4rem !important; }
h2 { font-size: 1.4rem !important; border-bottom: 1px solid var(--border); padding-bottom: 6px; }

/* Dividers */
hr { border-color: var(--border) !important; }

/* Selectbox / inputs */
div[data-baseweb="select"] > div {
    background: var(--panel) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}

/* Tables */
.stDataFrame { border: 1px solid var(--border) !important; }
.stDataFrame thead th {
    background: var(--panel) !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 2px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
}

/* Alert / info boxes */
.stAlert {
    border-left: 3px solid var(--accent) !important;
    background: rgba(0,229,255,0.05) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* Live badge */
.live-badge {
    display: inline-block;
    background: var(--danger);
    color: #fff;
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    padding: 2px 8px;
    border-radius: 2px;
    animation: blink 1.2s step-end infinite;
    vertical-align: middle;
    margin-left: 10px;
}
@keyframes blink { 50% { opacity: 0; } }

/* Score gauge container */
.gauge-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: var(--subtext);
    text-transform: uppercase;
    text-align: center;
    margin-top: -8px;
}
</style>
""", unsafe_allow_html=True)


# ─── Data loaders ─────────────────────────────────────────────────────────────
# Always resolve paths relative to THIS file's directory so the dashboard
# finds data no matter which folder `streamlit run` is launched from.
_HERE         = Path(__file__).parent.resolve()
LOG_FILE      = str(_HERE / "session_log.csv")
PROFILES_FILE = str(_HERE / "driver_profiles.json")
REPORTS_DIR   = _HERE / "reports"
SCREENSHOTS   = _HERE / "alerts"

PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono, monospace", color="#c8d0dc", size=11),
        xaxis=dict(gridcolor="#1e2530", linecolor="#1e2530", zerolinecolor="#1e2530"),
        yaxis=dict(gridcolor="#1e2530", linecolor="#1e2530", zerolinecolor="#1e2530"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2530"),
    )
)

@st.cache_data(ttl=3)
def load_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        # on_bad_lines="warn" skips broken rows instead of crashing;
        # engine="python" is more tolerant of extra commas in the details column.
        df = pd.read_csv(
            LOG_FILE,
            names=["timestamp", "event", "drowsiness_score", "driver", "details"],
            header=0,
            on_bad_lines="warn",
            engine="python",
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["drowsiness_score"] = pd.to_numeric(df["drowsiness_score"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        return df
    except Exception as e:
        st.warning(f"Could not parse {LOG_FILE}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_profiles():
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE) as f:
        return json.load(f)

@st.cache_data(ttl=10)
def load_reports():
    if not REPORTS_DIR.exists():
        return []
    files = sorted(REPORTS_DIR.glob("*.json"), reverse=True)
    reports = []
    for fp in files:
        with open(fp) as f:
            reports.append(json.load(f))
    return reports

def is_session_live(df):
    """True if there's a SESSION_START with no subsequent SESSION_END."""
    if df.empty:
        return False
    starts = df[df["event"] == "SESSION_START"]
    ends   = df[df["event"] == "SESSION_END"]
    if starts.empty:
        return False
    last_start = starts["timestamp"].max()
    if ends.empty:
        return True
    return last_start > ends["timestamp"].max()

def score_color(score):
    if score >= 70: return "#ff2d2d"
    if score >= 40: return "#ff8c00"
    if score >= 15: return "#00e5ff"
    return "#00e676"

def rating_stars(rating_str):
    """Extract star emoji from rating string."""
    for prefix in ["★★★★★", "★★★★☆", "★★★☆☆", "★★☆☆☆", "★☆☆☆☆"]:
        if rating_str.startswith(prefix):
            return prefix, rating_str[len(prefix):].strip()
    return "", rating_str


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Barlow Condensed",sans-serif;font-weight:900;
                font-size:1.6rem;letter-spacing:0.08em;color:#fff;
                text-transform:uppercase;padding:8px 0 4px 0;'>
        🚗 DrowseSense
    </div>
    <div style='font-family:"Share Tech Mono",monospace;font-size:0.65rem;
                color:#5a6478;letter-spacing:0.15em;margin-bottom:16px;'>
        ANALYTICS DASHBOARD v2.0
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigate",
                    ["📡 Live Monitor", "📊 Session History",
                     "👤 Driver Profiles", "📁 Reports Archive"],
                    label_visibility="collapsed")
    st.markdown("---")

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (3 s)", value=True)
    if auto_refresh:
        st.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
            'color:#5a6478;">Refreshing every 3 s</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    log_exists  = os.path.exists(LOG_FILE)
    prof_exists = os.path.exists(PROFILES_FILE)
    log_icon    = '✔' if log_exists  else '✘'
    prof_icon   = '✔' if prof_exists else '✘'
    log_col     = '#00e676' if log_exists  else '#ff2d2d'
    prof_col    = '#00e676' if prof_exists else '#ff2d2d'
    st.markdown(
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.58rem;'
        f'color:#5a6478;line-height:1.9;">'
        f'<b style="color:#3a4050;">DATA SOURCES</b><br>'
        f'<span style="color:{log_col};">{log_icon}</span> session_log.csv<br>'
        f'<span style="color:{prof_col};">{prof_icon}</span> driver_profiles.json<br>'
        f'<span style="color:#2a3040;font-size:0.5rem;">{str(_HERE)}</span>'
        f'</div>',
        unsafe_allow_html=True)


# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(0.1)
    st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "📡 Live Monitor":
    df = load_log()
    live = is_session_live(df)

    badge = '<span class="live-badge">● LIVE</span>' if live else \
            '<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;' \
            'color:#2a3040;padding:2px 8px;border:1px solid #1e2530;border-radius:2px;">' \
            '○ OFFLINE</span>'

    st.markdown(f'<h1>Live Monitor {badge}</h1>', unsafe_allow_html=True)

    if df.empty:
        st.info("No session data found. Start the detection script to begin logging.")
        st.stop()

    # Filter to current session
    starts = df[df["event"] == "SESSION_START"]
    if not starts.empty:
        session_start_ts = starts["timestamp"].max()
        sess_df = df[df["timestamp"] >= session_start_ts].copy()
    else:
        sess_df = df.copy()

    events_df = sess_df[~sess_df["event"].isin(["SESSION_START", "SESSION_END"])]
    driver    = sess_df["driver"].dropna().iloc[-1] if "driver" in sess_df.columns and not sess_df["driver"].dropna().empty else "Unknown"
    last_score = events_df["drowsiness_score"].dropna().iloc[-1] if not events_df.empty else 0

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    duration = (datetime.datetime.now() - session_start_ts.to_pydatetime()).total_seconds() / 60 if not starts.empty else 0
    with c1: st.metric("Driver",       driver)
    with c2: st.metric("Session Time", f"{int(duration):02d} min")
    with c3: st.metric("Fatigue Score", f"{last_score:.0f}/100")
    with c4: st.metric("Drowsy Alerts",
                        len(events_df[events_df["event"] == "DROWSY"]))
    with c5: st.metric("Phone Alerts",
                        len(events_df[events_df["event"] == "PHONE_DISTRACTION"]))

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("## Fatigue Score — Live Timeline")
        if not events_df.empty and "drowsiness_score" in events_df.columns:
            timeline = events_df[["timestamp", "drowsiness_score", "event"]].dropna()
            fig = go.Figure(layout=PLOT_TEMPLATE["layout"])
            # Shaded risk zones
            fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,45,45,0.08)",  line_width=0, annotation_text="CRITICAL", annotation_position="top right", annotation_font=dict(color="#ff2d2d", size=9))
            fig.add_hrect(y0=40, y1=70,  fillcolor="rgba(255,140,0,0.06)",  line_width=0, annotation_text="WARNING",  annotation_position="top right", annotation_font=dict(color="#ff8c00", size=9))
            fig.add_hrect(y0=15, y1=40,  fillcolor="rgba(0,229,255,0.04)",  line_width=0, annotation_text="MILD",     annotation_position="top right", annotation_font=dict(color="#00e5ff", size=9))
            # Score line
            fig.add_trace(go.Scatter(
                x=timeline["timestamp"], y=timeline["drowsiness_score"],
                mode="lines", line=dict(color="#00e5ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
                name="Score",
            ))
            # Event markers
            for evt, color, sym in [
                ("DROWSY",            "#ff2d2d", "circle"),
                ("YAWN",              "#ff8c00", "diamond"),
                ("HEAD_MOVEMENT",     "#ffe066", "triangle-up"),
                ("PHONE_DISTRACTION", "#bf5fff", "square"),
            ]:
                sub = timeline[timeline["event"] == evt]
                if not sub.empty:
                    fig.add_trace(go.Scatter(
                        x=sub["timestamp"], y=sub["drowsiness_score"],
                        mode="markers",
                        marker=dict(color=color, size=9, symbol=sym,
                                    line=dict(color="#fff", width=1)),
                        name=evt.replace("_", " ").title(),
                    ))
            fig.update_yaxes(range=[0, 105], title="Score")
            fig.update_layout(height=300, showlegend=True,
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True, key="live_timeline")
        else:
            st.caption("No events recorded yet in this session.")

    with col_right:
        st.markdown("## Risk Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(last_score),
            number=dict(font=dict(family="Barlow Condensed", size=36, color="#fff")),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color="#5a6478", size=9)),
                bar=dict(color=score_color(float(last_score)), thickness=0.3),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 15],  color="rgba(0,230,118,0.12)"),
                    dict(range=[15, 40], color="rgba(0,229,255,0.10)"),
                    dict(range=[40, 70], color="rgba(255,140,0,0.10)"),
                    dict(range=[70, 100],color="rgba(255,45,45,0.12)"),
                ],
                threshold=dict(line=dict(color="#fff", width=2), value=float(last_score)),
            ),
        ), layout=PLOT_TEMPLATE["layout"])
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True, key="live_gauge")

        # Alert breakdown donut
        st.markdown("## Alert Breakdown")
        counts = {
            "Drowsy": len(events_df[events_df["event"] == "DROWSY"]),
            "Yawn":   len(events_df[events_df["event"] == "YAWN"]),
            "Head":   len(events_df[events_df["event"] == "HEAD_MOVEMENT"]),
            "Phone":  len(events_df[events_df["event"] == "PHONE_DISTRACTION"]),
        }
        if sum(counts.values()) > 0:
            fig_pie = go.Figure(go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.6,
                marker=dict(colors=["#ff2d2d", "#ff8c00", "#ffe066", "#bf5fff"],
                            line=dict(color="#090c10", width=2)),
                textinfo="label+percent",
                textfont=dict(family="Share Tech Mono", size=10),
            ), layout=PLOT_TEMPLATE["layout"])
            fig_pie.update_layout(height=220, showlegend=False,
                                  margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_pie, use_container_width=True, key="live_pie")
        else:
            st.caption("No alerts yet.")

    # ── Recent events table ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Recent Events")
    if not events_df.empty:
        recent = events_df.tail(15)[["timestamp", "event", "drowsiness_score", "driver"]].copy()
        recent["timestamp"] = recent["timestamp"].dt.strftime("%H:%M:%S")
        recent.columns = ["Time", "Event", "Score", "Driver"]
        st.dataframe(recent[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("No events yet.")

    if auto_refresh:
        time.sleep(3)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Session History":
    st.markdown("<h1>Session History</h1>", unsafe_allow_html=True)
    df = load_log()
    if df.empty:
        st.info("No session data yet.")
        st.stop()

    # Identify session boundaries
    starts = df[df["event"] == "SESSION_START"].copy()
    ends   = df[df["event"] == "SESSION_END"].copy()

    if starts.empty:
        st.info("No sessions found.")
        st.stop()

    events_only = df[~df["event"].isin(["SESSION_START", "SESSION_END"])].copy()

    # ── Aggregate per-session stats ────────────────────────────────────────────
    sessions = []
    for idx, row in starts.iterrows():
        s_time = row["timestamp"]
        # Find matching end
        later_ends = ends[ends["timestamp"] > s_time]
        e_time = later_ends["timestamp"].min() if not later_ends.empty else None
        seg = events_only[events_only["timestamp"] >= s_time]
        if e_time:
            seg = seg[seg["timestamp"] <= e_time]
        sessions.append({
            "Session Start": s_time.strftime("%Y-%m-%d %H:%M"),
            "Driver":        row.get("driver", "—"),
            "Duration (min)": round((e_time - s_time).total_seconds() / 60, 1) if e_time else "—",
            "Drowsy":  len(seg[seg["event"] == "DROWSY"]),
            "Yawn":    len(seg[seg["event"] == "YAWN"]),
            "Head":    len(seg[seg["event"] == "HEAD_MOVEMENT"]),
            "Phone":   len(seg[seg["event"] == "PHONE_DISTRACTION"]),
            "Peak Score": round(seg["drowsiness_score"].max(), 1) if not seg.empty else 0,
        })

    sess_summary = pd.DataFrame(sessions)
    st.dataframe(sess_summary, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Trend charts ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Alert Frequency Over Sessions")
        if len(sessions) > 1:
            trend = pd.DataFrame(sessions)
            fig = go.Figure(layout=PLOT_TEMPLATE["layout"])
            for col_name, color in [("Drowsy","#ff2d2d"),("Yawn","#ff8c00"),
                                     ("Head","#ffe066"),("Phone","#bf5fff")]:
                fig.add_trace(go.Bar(
                    x=trend["Session Start"], y=trend[col_name],
                    name=col_name, marker_color=color, opacity=0.85,
                ))
            fig.update_layout(barmode="stack", height=300,
                              xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, key="hist_alert_bar")
        else:
            st.caption("Need 2+ sessions for trend data.")

    with col2:
        st.markdown("## Peak Fatigue Score per Session")
        if len(sessions) > 0:
            trend = pd.DataFrame(sessions)
            numeric_peak = pd.to_numeric(trend["Peak Score"], errors="coerce").fillna(0)
            colors = [score_color(v) for v in numeric_peak]
            fig = go.Figure(layout=PLOT_TEMPLATE["layout"])
            fig.add_trace(go.Bar(
                x=trend["Session Start"],
                y=numeric_peak,
                marker_color=colors,
                name="Peak Score",
            ))
            fig.add_hline(y=70, line_dash="dot", line_color="#ff2d2d",
                          annotation_text="Critical", annotation_font_color="#ff2d2d")
            fig.add_hline(y=40, line_dash="dot", line_color="#ff8c00",
                          annotation_text="Warning",  annotation_font_color="#ff8c00")
            fig.update_layout(height=300, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, key="hist_peak_bar")

    # ── Hour-of-day heatmap ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Incident Heatmap — Hour of Day")
    if not events_only.empty:
        events_only["hour"] = events_only["timestamp"].dt.hour
        events_only["day"]  = events_only["timestamp"].dt.strftime("%a")
        pivot = events_only.groupby(["day", "hour"]).size().unstack(fill_value=0)
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[0,"#0f1318"],[0.4,"#004080"],[0.7,"#ff8c00"],[1,"#ff2d2d"]],
            showscale=True,
        ), layout=PLOT_TEMPLATE["layout"])
        fig_heat.update_layout(height=220)
        st.plotly_chart(fig_heat, use_container_width=True, key="hist_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DRIVER PROFILES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Driver Profiles":
    st.markdown("<h1>Driver Profiles</h1>", unsafe_allow_html=True)
    profiles = load_profiles()

    if not profiles:
        st.info("No profiles found. Start a session from the detection script to create one.")
        st.stop()

    names = sorted(profiles.keys())
    selected = st.selectbox("Select Driver", names, label_visibility="visible")

    if selected:
        p = profiles[selected]
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Sessions",    p.get("total_sessions", 0))
        with c2: st.metric("Total Drive Time",  f"{p.get('total_drive_minutes', 0):.0f} min")
        with c3: st.metric("Drowsy Alerts",     p.get("total_drowsy_alerts", 0))
        with c4: st.metric("Phone Alerts",      p.get("total_phone_alerts", 0))

        c5, c6, c7, c8 = st.columns(4)
        with c5: st.metric("Yawn Alerts",       p.get("total_yawn_alerts", 0))
        with c6: st.metric("Head Alerts",       p.get("total_head_alerts", 0))
        with c7:
            total = (p.get("total_drowsy_alerts", 0) + p.get("total_yawn_alerts", 0)
                     + p.get("total_head_alerts", 0) + p.get("total_phone_alerts", 0))
            sess  = max(p.get("total_sessions", 1), 1)
            st.metric("Alerts / Session", f"{total/sess:.1f}")
        with c8:
            drive = p.get("total_drive_minutes", 0)
            rate  = (total / drive * 60) if drive > 0 else 0
            st.metric("Alerts / Hour", f"{rate:.1f}")

        st.markdown("---")
        st.markdown("## Alert Profile")

        col_l, col_r = st.columns([1, 1])
        with col_l:
            alert_data = {
                "Type":  ["Drowsy", "Yawn", "Head", "Phone"],
                "Count": [p.get("total_drowsy_alerts", 0),
                          p.get("total_yawn_alerts", 0),
                          p.get("total_head_alerts", 0),
                          p.get("total_phone_alerts", 0)],
                "Color": ["#ff2d2d", "#ff8c00", "#ffe066", "#bf5fff"],
            }
            fig = go.Figure(go.Bar(
                x=alert_data["Type"], y=alert_data["Count"],
                marker_color=alert_data["Color"],
            ), layout=PLOT_TEMPLATE["layout"])
            fig.update_layout(height=260, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="profile_alert_bar")

        with col_r:
            # Radar chart
            categories = ["Drowsy", "Yawn", "Head Move", "Phone"]
            vals = [p.get("total_drowsy_alerts", 0),
                    p.get("total_yawn_alerts", 0),
                    p.get("total_head_alerts", 0),
                    p.get("total_phone_alerts", 0)]
            max_val = max(vals) if max(vals) > 0 else 1
            norm = [v / max_val for v in vals]

            fig_r = go.Figure(go.Scatterpolar(
                r=norm + [norm[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(0,229,255,0.12)",
                line=dict(color="#00e5ff", width=2),
            ), layout=PLOT_TEMPLATE["layout"])
            fig_r.update_layout(
                height=260,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=False, range=[0, 1]),
                    angularaxis=dict(color="#5a6478", gridcolor="#1e2530"),
                ),
            )
            st.plotly_chart(fig_r, use_container_width=True, key="profile_radar")

        # Compare all drivers
        st.markdown("---")
        st.markdown("## All Drivers Comparison")
        rows = []
        for name, data in profiles.items():
            total_a = (data.get("total_drowsy_alerts", 0)
                       + data.get("total_yawn_alerts", 0)
                       + data.get("total_head_alerts", 0)
                       + data.get("total_phone_alerts", 0))
            rows.append({"Driver": name,
                         "Sessions": data.get("total_sessions", 0),
                         "Drive Time (min)": data.get("total_drive_minutes", 0),
                         "Total Alerts": total_a})
        compare_df = pd.DataFrame(rows).sort_values("Total Alerts", ascending=False)

        fig_cmp = px.bar(compare_df, x="Driver", y="Total Alerts",
                         color="Total Alerts",
                         color_continuous_scale=["#00e676", "#00e5ff", "#ff8c00", "#ff2d2d"])
        fig_cmp.update_layout(**PLOT_TEMPLATE["layout"], height=260,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_cmp, use_container_width=True, key="profile_compare")
        st.dataframe(compare_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — REPORTS ARCHIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Reports Archive":
    st.markdown("<h1>Reports Archive</h1>", unsafe_allow_html=True)
    reports = load_reports()

    if not reports:
        st.info("No session reports found. Press **R** in the detection window to export one.")
        st.stop()

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        all_drivers = sorted(set(r.get("driver", "Unknown") for r in reports))
        filter_driver = st.selectbox("Filter by Driver", ["All"] + all_drivers)
    with col_f2:
        filter_risk = st.selectbox("Filter by Risk Level",
                                   ["All", "ALERT", "MILD", "WARNING", "CRITICAL"])

    filtered = [r for r in reports
                if (filter_driver == "All" or r.get("driver") == filter_driver)
                and (filter_risk == "All" or r.get("risk_level") == filter_risk)]

    st.markdown(f"**{len(filtered)}** report(s) found")
    st.markdown("---")

    for _report_idx, r in enumerate(filtered):
        stars, label = rating_stars(r.get("safety_rating", ""))
        risk   = r.get("risk_level", "—")
        rcolor = {"ALERT":"#00e676","MILD":"#00e5ff",
                  "WARNING":"#ff8c00","CRITICAL":"#ff2d2d"}.get(risk, "#fff")

        with st.expander(
            f"🗓 {r.get('date','?')}  ·  "
            f"👤 {r.get('driver','?')}  ·  "
            f"⏱ {r.get('duration_seconds',0)//60} min  ·  "
            f"Score: {r.get('final_score','?')}  ·  "
            f"{stars} {label}"
        ):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("Drowsy", r.get("alerts", {}).get("drowsy", 0))
            with c2: st.metric("Yawn",   r.get("alerts", {}).get("yawn", 0))
            with c3: st.metric("Head",   r.get("alerts", {}).get("head", 0))
            with c4: st.metric("Phone",  r.get("alerts", {}).get("phone", 0))
            with c5:
                st.markdown