"""Data loading for the VIX vol-carry engine.

VIX spot indices (.VIX, .VIX3M, .VIX9D, .VVIX), VIX futures continuation
(VXc1..VXc3), and SPY. LSEG data, 2010-2026.
"""
from pathlib import Path
import pandas as pd

SERIES = {"VXc1": "VXc1", "VXc2": "VXc2", "VXc3": "VXc3",
          "VIX": "_VIX", "VIX3M": "_VIX3M", "VIX9D": "_VIX9D", "VVIX": "_VVIX",
          "SPY": "SPY"}


def load(data_dir: Path) -> pd.DataFrame:
    raw = Path(data_dir) / "raw_prices"
    cols = {}
    for name, fname in SERIES.items():
        s = pd.read_csv(raw / f"{fname}.csv", parse_dates=["Date"], index_col="Date").iloc[:, 0]
        cols[name] = s.astype(float).sort_index()
    df = pd.concat(cols, axis=1)
    # VIX futures start a touch later; keep rows where the core (VXc1, VIX, SPY) exist
    df = df.dropna(subset=["VXc1", "VXc2", "VIX", "VIX3M", "SPY"])
    df.index.name = "Date"
    return df


def fetch_from_lseg(data_dir: Path, start="2010-01-01", end="2026-06-05") -> None:
    """Refresh CSVs from LSEG (active Workspace session required)."""
    import lseg.data as ld
    raw = Path(data_dir) / "raw_prices"; raw.mkdir(parents=True, exist_ok=True)
    rics = {"VXc1": "VXc1", "VXc2": "VXc2", "VXc3": "VXc3", "_VIX": ".VIX",
            "_VIX3M": ".VIX3M", "_VIX9D": ".VIX9D", "_VVIX": ".VVIX", "SPY": "SPY"}
    ld.open_session()
    try:
        for fname, ric in rics.items():
            df = ld.get_history(universe=ric, fields=["TRDPRC_1"], interval="daily",
                                start=start, end=end)
            s = df.iloc[:, 0].dropna(); s.index = pd.to_datetime(s.index)
            s.sort_index().to_csv(raw / f"{fname}.csv", header=["TRDPRC_1"], index_label="Date")
            print(f"  saved {fname}: {len(s)} obs")
    finally:
        ld.close_session()
