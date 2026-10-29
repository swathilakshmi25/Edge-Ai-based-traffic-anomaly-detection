"""EDGE AI BASED TRAFFIC ANOMALY DETECTION - Streamlit entry point.

Run:  python -m streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="EDGE AI BASED TRAFFIC ANOMALY DETECTION", page_icon="🚦", layout="wide")

from traffic_anomaly.config import ConfigError  # noqa: E402
from traffic_anomaly.dashboard import pages  # noqa: E402
from traffic_anomaly.dashboard.helpers import get_config, get_store  # noqa: E402
from traffic_anomaly.dashboard.style import CSS  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

try:
    cfg = get_config()
except ConfigError as exc:
    st.error(f"Configuration file problem:\n\n{exc}")
    st.stop()
store = get_store(cfg)

PAGES = {
    "Dashboard": pages.page_dashboard,
    "Live Monitoring / Video Analysis": pages.page_video_analysis,
    "Anomaly Detection": pages.page_anomaly,
    "Alert Management": pages.page_alerts,
    "Analytics and Reports": pages.page_analytics,
    "Settings": pages.page_settings,
}

with st.sidebar:
    st.markdown("### 🚦 Navigation")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    view = st.selectbox("Data view", ["Real video (AI detector)", "Simulated demo only", "All records"],
                        help="Real and simulated records are stored separately and never mixed unless you choose 'All'.")
    mode = {"Real video (AI detector)": "real", "Simulated demo only": "demo", "All records": None}[view]
    st.caption("Local processing only. Alerts are potential anomalies requiring human review.")

st.markdown("<div class='app-title'>EDGE AI BASED TRAFFIC ANOMALY DETECTION</div>"
            "<div class='app-sub'>Local video analysis &bull; vehicle tracking &bull; rule-based alerts</div>",
            unsafe_allow_html=True)
PAGES[page](cfg, store, mode)
