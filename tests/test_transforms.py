import numpy as np

from chap_ar.transforms import ZScaler, period_start_date, year_position_from_period


def test_zscaler_standardizes_first_element():
    mu = np.array([1.0, 2.0])
    std = np.array([2.0, 4.0])
    feats = np.array([[3.0, 6.0]])
    out = ZScaler(mu, std)((feats,))
    assert np.allclose(out[0], (feats - mu) / std)


def test_year_position_bounds():
    assert year_position_from_period("2024-01") < 0.02
    assert year_position_from_period("2024-12") > 0.9


def test_period_start_date_monthly_and_weekly():
    assert period_start_date("2010-03").month == 3
    weekly = period_start_date("2003-12-29/2004-01-04")
    assert (weekly.year, weekly.month, weekly.day) == (2003, 12, 29)
