"""Offline unit tests for the dashboard's pure helpers.

No test here touches yfinance: every fetch-dependent test monkeypatches
D._get_frame with a stub. CI's dashboard health job is the integration
gate for real data.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import common
from stock_analyst_mcp import data as D


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_caches():
    common.load_full_frame.clear()
    common.patterns_for.clear()
    yield
    common.load_full_frame.clear()
    common.patterns_for.clear()


def make_frame(dates: list[str], close: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    df = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "CloseAdj": close},
        index=idx,
    )
    df.attrs["rows_used"] = len(df)
    return df


# ---------------------------------------------------------------------------
# parse_view_params: query-param validation / clamping
# ---------------------------------------------------------------------------

class TestParseViewParams:
    def test_good_params_pass_through(self):
        out = common.parse_view_params(
            {"symbol": "TCS.NS", "as_of": "2024-11-05", "lookback": "250"})
        assert out == {"symbol": "TCS.NS", "as_of": date(2024, 11, 5),
                       "lookback": 250}

    def test_bad_date_falls_back_to_today_minus_60(self):
        out = common.parse_view_params({"as_of": "not-a-date"})
        assert out["as_of"] == common.IST_TODAY - timedelta(days=60)

    def test_empty_date_falls_back(self):
        out = common.parse_view_params({})
        assert out["as_of"] == common.IST_TODAY - timedelta(days=60)
        assert out["symbol"] == "RELIANCE.NS"
        assert out["lookback"] == 120

    def test_future_date_falls_back(self):
        out = common.parse_view_params(
            {"as_of": str(common.IST_TODAY + timedelta(days=30))})
        assert out["as_of"] == common.IST_TODAY - timedelta(days=60)

    def test_lookback_clamped_to_10_400(self):
        assert common.parse_view_params({"lookback": "5"})["lookback"] == 10
        assert common.parse_view_params({"lookback": "999"})["lookback"] == 400
        assert common.parse_view_params({"lookback": "120"})["lookback"] == 120

    def test_bad_lookback_falls_back_to_default(self):
        assert common.parse_view_params({"lookback": "abc"})["lookback"] == 120
        assert common.parse_view_params({"lookback": ""})["lookback"] == 120

    def test_empty_symbol_falls_back_to_default(self):
        assert common.parse_view_params({"symbol": "  "})["symbol"] == "RELIANCE.NS"


# ---------------------------------------------------------------------------
# slice_lookback: in-memory tail, never mutates the cached frame
# ---------------------------------------------------------------------------

class TestSliceLookback:
    def test_tail_length(self):
        full = make_frame([f"2024-01-{d:02d}" for d in range(1, 11)],
                          [10.0] * 10)
        view = common.slice_lookback(full, 4)
        assert len(view) == 4
        assert view.index[-1] == full.index[-1]

    def test_slice_is_a_copy_not_the_cache(self):
        full = make_frame(["2024-01-01", "2024-01-02"], [10.0, 11.0])
        view = common.slice_lookback(full, 2)
        view["CloseAdj"] = -1.0  # mutating the view
        assert float(full["CloseAdj"].iloc[-1]) == 11.0  # cache unchanged

    def test_rows_used_preserved_through_slice(self):
        full = make_frame([f"2024-01-{d:02d}" for d in range(1, 6)],
                          [1.0] * 5)
        full.attrs["rows_used"] = 42  # pre-trim provenance from the server
        view = common.slice_lookback(full, 2)
        assert view.attrs["rows_used"] == 42


# ---------------------------------------------------------------------------
# within_as_of: the point-in-time guarantee (explicit check, never a bare assert)
# ---------------------------------------------------------------------------

class TestWithinAsOf:
    def test_rejects_future_row(self):
        df = make_frame(["2024-11-04", "2024-11-06"], [1.0, 2.0])
        assert common.within_as_of(df, date(2024, 11, 5)) is False

    def test_accepts_row_on_as_of(self):
        df = make_frame(["2024-11-04", "2024-11-05"], [1.0, 2.0])
        assert common.within_as_of(df, date(2024, 11, 5)) is True


# ---------------------------------------------------------------------------
# vintage_delta: join + delta maths
# ---------------------------------------------------------------------------

class TestVintageDelta:
    def test_delta_maths_against_hand_computed_pair(self):
        idx = pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02",
                                               "2024-01-03"]))
        v_then = pd.Series([100.0, 200.0, 300.0], index=idx)
        v_now = pd.Series([110.0, 190.0, 330.0], index=idx)
        out = common.vintage_delta(v_then, v_now)
        assert list(out["delta_pct"].round(6)) == [10.0, -5.0, 10.0]

    def test_non_overlapping_indexes_return_empty_not_raise(self):
        a = pd.Series([1.0], index=pd.DatetimeIndex(["2024-01-01"]))
        b = pd.Series([2.0], index=pd.DatetimeIndex(["2024-06-01"]))
        out = common.vintage_delta(a, b)
        assert isinstance(out, pd.DataFrame)
        assert out.empty
        assert list(out.columns) == ["then", "now", "delta_pct"]

    def test_join_keeps_common_dates_only(self):
        idx1 = pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        idx2 = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]))
        out = common.vintage_delta(pd.Series([1.0, 2.0], index=idx1),
                                   pd.Series([3.0, 9.0], index=idx2))
        assert list(out.index) == list(pd.DatetimeIndex(["2024-01-02"]))


# ---------------------------------------------------------------------------
# rank_peers: descending value, None last, explicit rank
# ---------------------------------------------------------------------------

class TestRankPeers:
    def test_descending_and_none_last(self):
        rows = [{"symbol": "A", "value": 0.01},
                {"symbol": "B", "value": 0.05},
                {"symbol": "C", "value": None},
                {"symbol": "D", "value": 0.03}]
        ranked = common.rank_peers(rows, "return_20d")
        assert [r["symbol"] for r in ranked] == ["B", "D", "A", "C"]
        assert [r["rank"] for r in ranked] == [1, 2, 3, 4]

    def test_keeps_other_fields(self):
        rows = [{"symbol": "A", "value": 1.0, "current_price": 10.0,
                 "currency": "INR"}]
        ranked = common.rank_peers(rows, "vol_60d")
        assert ranked[0]["current_price"] == 10.0
        assert ranked[0]["rank"] == 1


# ---------------------------------------------------------------------------
# private-API pin + single-fetch property
# ---------------------------------------------------------------------------

class TestPrivateApiPin:
    def test_server_still_exposes_the_private_function_the_dashboard_uses(self):
        assert hasattr(D, "_get_frame")

    def test_pin_is_declared_loudly(self):
        assert "_get_frame" in common.PRIVATE_API_USED


class TestSingleFetch:
    def test_one_get_frame_call_per_symbol_as_of(self, monkeypatch):
        calls = []
        frame = make_frame(["2024-11-01", "2024-11-04", "2024-11-05"],
                           [10.0, 11.0, 12.0])

        def fake_get_frame(symbol, as_of, lookback_days=90):
            calls.append((symbol, as_of, lookback_days))
            return frame.copy(), date(2024, 11, 5), [], []

        monkeypatch.setattr(D, "_get_frame", fake_get_frame)

        common.load_full_frame("STUB.NS", "2024-11-05")   # first call: miss
        common.load_full_frame("STUB.NS", "2024-11-05")   # second: cache hit
        common.patterns_for("STUB.NS", "2024-11-05", 3, 20)  # cache hit again

        assert len(calls) == 1
        assert calls[0] == ("STUB.NS", "2024-11-05", 100000)

    def test_distinct_as_of_fetches_again(self, monkeypatch):
        calls = []
        frame = make_frame(["2024-11-01", "2024-11-04", "2024-11-05"],
                           [10.0, 11.0, 12.0])

        def fake_get_frame(symbol, as_of, lookback_days=90):
            calls.append((symbol, as_of))
            return frame.copy(), date(2024, 11, 5), [], []

        monkeypatch.setattr(D, "_get_frame", fake_get_frame)

        common.load_full_frame("STUB2.NS", "2024-11-05")
        common.load_full_frame("STUB2.NS", "2024-11-04")
        assert len(calls) == 2
