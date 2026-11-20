"""Overview page: price chart + indicator overlays + returns strip.

Merges the three original pages (price / indicators / returns):
- candlestick (or line) with pattern markers and support/resistance,
- indicator overlays (RSI / MACD get their own subplot row),
- the honest trend badge,
- a 1/5/20/60-day simple-return strip,
- the full indicator table in an expander.

Overlay series mirror the server's technicals.py math exactly
(rolling/ewm on CloseAdj); VWAP is intentionally not offered as an
overlay — the table's VWAP is one whole-window volume-weighted number,
not a time series.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st

import common
from stock_analyst_mcp import technicals  # noqa: E402  (path shim in common)

# Overlays that have a well-defined per-date series. VWAP is excluded: the
# server computes it as a single whole-window volume-weighted average.
OVERLAYABLE = [i for i in common.ALL_IND if i != "VWAP"]
_SUBPLOT_INDICATORS = ("RSI_14", "MACD")


def overlay_series(df: pd.DataFrame, ind: str) -> dict[str, pd.Series]:
    """Per-date indicator series for the chart, mirroring the server's
    technicals.compute_all math (rolling/ewm on CloseAdj)."""
    close = df["CloseAdj"].astype(float)
    out: dict[str, pd.Series] = {}
    if ind == "SMA_20":
        out["SMA_20"] = close.rolling(20).mean()
    elif ind == "SMA_50":
        out["SMA_50"] = close.rolling(50).mean()
    elif ind == "SMA_200":
        out["SMA_200"] = close.rolling(200).mean()
    elif ind == "EMA_12":
        out["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    elif ind == "EMA_26":
        out["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    elif ind == "BOLLINGER" and len(close) >= 20:
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        out["BOLLINGER_UPPER"] = mid + 2 * std
        out["BOLLINGER_MID"] = mid
        out["BOLLINGER_LOWER"] = mid - 2 * std
    elif ind == "RSI_14":
        out["RSI_14"] = _rsi_series(close, 14)
    elif ind == "MACD":
        line = _macd_line(close)
        out["MACD"] = line
        out["MACD_SIGNAL"] = line.ewm(span=9, adjust=False).mean()
    elif ind == "ATR_14" and len(close) >= 15:
        high, low = df["High"].astype(float), df["Low"].astype(float)
        prev = close.shift(1)
        tr = pd.concat([(high - low), (high - prev).abs(),
                        (low - prev).abs()], axis=1).max(axis=1)
        out["ATR_14"] = tr.rolling(14).mean()
    return out


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(avg_loss != 0, 100.0)


def _macd_line(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return ema12 - ema26


data = common.get_data()
if data is None:
    st.stop()
symbol, as_of, lookback, df, df_full, close_adj, current_price = data
as_of_str = as_of.isoformat() if isinstance(as_of, date) else str(as_of)

st.subheader(f"Price — {symbol}")

show_line = st.checkbox("Show line chart", value=False)
overlays = st.multiselect("Indicator overlays", options=OVERLAYABLE,
                          default=["SMA_50"])

# Cached detector (window 20, same default the old price page used).
pres = common.patterns_for(symbol, as_of_str, lookback, 20)
if pres.get("error"):
    st.error(pres["error"])
pattern_list = pres.get("patterns", [])
support_resistance = pres.get("support_resistance", {})

sub_indicators = [i for i in overlays if i in _SUBPLOT_INDICATORS]
n_rows = 2 if sub_indicators else 1
fig = sp.make_subplots(
    rows=n_rows, cols=1, shared_xaxes=True,
    row_heights=([0.75, 0.25] if n_rows == 2 else None),
    vertical_spacing=0.05 if n_rows == 2 else 0,
)

if show_line:
    for column in ("CloseAdj", "Close"):
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode="lines",
                                 name=column), row=1, col=1)
else:
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"].astype(float),
            high=df["High"].astype(float),
            low=df["Low"].astype(float),
            close=df["CloseAdj"].astype(float),
            name="OHLC (CloseAdj)",
        ),
        row=1, col=1,
    )

for ind in overlays:
    for name, series in overlay_series(df, ind).items():
        row = 2 if ind in _SUBPLOT_INDICATORS else 1
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines",
                                 name=name), row=row, col=1)

if pattern_list:
    marked = [p for p in pattern_list
              if p.get("date") in {str(idx.date()) for idx in df.index}]
    if marked:
        fig.add_scatter(
            x=[p["date"] for p in marked],
            y=[float(df.loc[df.index.strftime("%Y-%m-%d") == p["date"],
                            "Low"].iloc[0])
               for p in marked],
            mode="markers+text",
            text=[p["name"] for p in marked],
            textposition="bottom center",
            marker={"size": 10, "symbol": "diamond"},
            name="pattern",
            row=1, col=1,
        )
for level in support_resistance.get("support", []):
    fig.add_hline(y=level, line_dash="dot", line_color="green", row=1, col=1)
for level in support_resistance.get("resistance", []):
    fig.add_hline(y=level, line_dash="dot", line_color="red", row=1, col=1)

fig.update_layout(title=f"{symbol} OHLC (adjusted close)",
                  xaxis_title="date", yaxis_title="INR", height=600)
st.plotly_chart(fig, width="stretch")
st.caption("Blue diamonds mark candlestick patterns from the last 20 bars; "
           "green dotted = support, red dotted = resistance.")

# ---- trend badge (unchanged logic and messages) ----
calc = technicals.compute_all(df, common.ALL_IND)
sma50, sma200 = calc.get("SMA_50"), calc.get("SMA_200")
if sma200 is None:
    trend, trend_note = "insufficient data (need lookback ≥200 for SMA_200)", ""
elif sma50 is None:
    trend, trend_note = "insufficient data (need lookback ≥50 for SMA_50)", ""
else:
    trend = "bullish" if sma50 > sma200 else "bearish"
    trend_note = f" (SMA_50 {'>' if trend == 'bullish' else '≤'} SMA_200)"
tc = st.columns(2)
tc[0].metric("current_price", f"{current_price} INR")
tc[1].metric("trend", trend, trend_note or None)

# ---- returns strip: 1/5/20/60-day simple % as st.metric deltas ----
def simple_return(close: pd.Series, w: int) -> float | None:
    if len(close) <= w or close.iloc[-1 - w] == 0:
        return None
    past, last = close.iloc[-1 - w], close.iloc[-1]
    return float((last - past) / past)


rc = st.columns(4)
for i, w in enumerate((1, 5, 20, 60)):
    r = simple_return(close_adj, w)
    rc[i].metric(f"{w}d", f"{current_price:,.2f}",
                 delta=f"{r * 100:+.2f}%" if r is not None else None,
                 delta_color="normal" if r is not None else "off")

with st.expander("Simple / log returns table"):
    rows = []
    for w in (1, 5, 20, 60):
        if len(close_adj) <= w or close_adj.iloc[-1 - w] == 0:
            rows.append({"window": f"{w}d", "simple": None, "log": None,
                         "pct_simple": None})
            continue
        past, last = close_adj.iloc[-1 - w], close_adj.iloc[-1]
        simple = (last - past) / past
        rows.append({"window": f"{w}d", "simple": round(float(simple), 6),
                     "log": round(float(np.log(last / past)), 6),
                     "pct_simple": round(float(simple) * 100, 4)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

with st.expander("All indicator values"):
    st.dataframe(pd.DataFrame([{"indicator": k, "value": v}
                               for k, v in calc.items()]), hide_index=True)
    st.caption("VWAP is one whole-window volume-weighted average price, "
               "not a time series, so it has no chart overlay.")
