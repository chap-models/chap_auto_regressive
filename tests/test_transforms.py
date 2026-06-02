from datetime import datetime

import numpy as np

from chap_ar.transforms import ZScaler, year_position_from_datetime


def test_zscaler_standardizes_first_element():
    mu = np.array([1.0, 2.0])
    std = np.array([2.0, 4.0])
    feats = np.array([[3.0, 6.0]])
    out = ZScaler(mu, std)((feats,))
    assert np.allclose(out[0], (feats - mu) / std)


def test_year_position_bounds():
    assert year_position_from_datetime(datetime(2024, 1, 1)) < 0.02
    assert year_position_from_datetime(datetime(2024, 12, 31)) > 0.99
