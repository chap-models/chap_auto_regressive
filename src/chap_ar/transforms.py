import dataclasses
from datetime import datetime

import numpy as np
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def get_series(data: DataSet[FullData]):
    x = []
    y = []
    for series in data.values():
        year_position = [year_position_from_datetime(period.start_timestamp.date) for period in series.time_period]
        x.append(np.array((series.rainfall, series.mean_temperature, series.population, year_position)).T)  # type: ignore
        if hasattr(series, "disease_cases"):
            y.append(series.disease_cases)
    assert not np.any(np.isnan(x))
    return np.array(x), np.array(y)


def year_position_from_datetime(dt: datetime) -> float:
    day = dt.timetuple().tm_yday
    return day / 365


def get_feature_normalizer(data_set, i=0):
    x = data_set.predictors(i)
    mu = np.mean(x, axis=(0, 1))
    std = np.std(x, axis=(0, 1))
    return lambda x: x[:i] + ((x[i] - mu) / std,) + x[i + 1 :]


@dataclasses.dataclass
class ZScaler:
    mu: np.ndarray
    std: np.ndarray

    def __call__(self, x):
        i = 0
        return x[:i] + ((x[i] - self.mu) / self.std,) + x[i + 1 :]

    @classmethod
    def from_data(cls, data_set):
        return ZScaler(np.mean(data_set.predictors(0), axis=(0, 1)), np.std(data_set.predictors(0), axis=(0, 1)))


class DataDependentTransform:
    def __init__(self, data_set):
        self.data_set = data_set

    def __call__(self, x) -> tuple: ...


class DataDependentNormalizer:
    def __init__(self, data_set):
        self.data_set = data_set

    def __call__(self, x) -> tuple: ...


def t_chain(normalizers):
    def t(x):
        for n in normalizers:
            x = n(x)
        return x

    return t
