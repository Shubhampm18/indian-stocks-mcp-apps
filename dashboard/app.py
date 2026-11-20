"""Streamlit multi-page dashboard for indian-stocks-mcp's data layer.

Entry point: defines the navigation (Overview / Patterns / Peers /
Vintage) and renders the title. Shared sidebar + guarded data fetch live
in common.py; each page file owns one section.

GUARANTEE: no row with date > as_of appears anywhere — get_data() in
common.py enforces it before any page renders content; the Vintage page
enforces it per vintage (each frame against its own as_of).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Stock Analyst Dashboard", layout="wide")
st.title("📈 Stock Analyst Dashboard — NSE / BSE (as_of point-in-time)")

page = st.navigation(
    [
        st.Page("pages/overview.py", title="Overview", icon="📈", default=True),
        st.Page("pages/patterns.py", title="Patterns", icon="🔍"),
        st.Page("pages/peers.py", title="Peers", icon="⚖️"),
        st.Page("pages/vintage.py", title="Vintage", icon="🕰️"),
    ]
)

import common  # noqa: E402  (after set_page_config)

common.render_sidebar_common()
page.run()
