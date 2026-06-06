"""Unit tests — run with: pytest -q"""
import sys
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load
from src.strategy import build, quant_checks

DATA = REPO / "data"


@pytest.fixture(scope="module")
def df():
    return load(DATA)


@pytest.fixture(scope="module")
def res(df):
    return build(df)


def test_core_no_nan(df):
    assert not df[["VXc1", "VXc2", "VIX", "VIX3M", "SPY"]].isna().any().any()
    assert df.index.is_monotonic_increasing and df.index.is_unique


def test_vrp_positive(res):
    # the variance risk premium should be positive on average
    assert res["vrp_mean"] > 0


def test_exposure_short_or_flat(res):
    # vol-carry only shorts or sits flat — never long vol
    assert (res["exposure"].dropna() <= 1e-9).all()
    assert res["exposure"].dropna().min() >= -1.0 - 1e-9


def test_signal_lagged(res):
    assert np.isnan(res["ret_carry"].iloc[0])


def test_crash_filter_flat_in_backwardation(df, res):
    # when VIX > VIX3M (backwardation), exposure must be 0
    bwd = df["VIX"] > df["VIX3M"]
    assert (res["exposure"][bwd].abs() < 1e-9).all()


def test_all_quant_checks_pass(df, res):
    assert all(p for _, p, _ in quant_checks(df, res))
