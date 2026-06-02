from typing import Iterable, Tuple

import numpy as np


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y):
    y = y.copy()
    for row in y:
        nans, x = nan_helper(row)
        row[nans] = np.interp(x(nans), x(~nans), row[~nans])
    return y


class DataSet:
    def __init__(self, X, y, forecast_length, context_length=None):
        self._X = X
        self._y = y
        self._context_length = context_length or X.shape[1] - forecast_length
        self._forecast_length = forecast_length
        self._total_length = self._context_length + forecast_length
        self._interpolated_y = interpolate_nans(y)
        self._transform = lambda x: x

    def set_transform(self, transform):
        self._transform = transform

    def predictors(self, i):
        return (self._X, self._interpolated_y, self._y)[i]

    def __len__(self):
        return self._X.shape[1] - self._total_length + 1

    def __getitem__(self, item):
        start = item
        return self._transform(
            (
                self._X[:, start : start + self._total_length],
                self._interpolated_y[:, start : start + self._context_length],
            )
        ) + (self._y[:, start : start + self._total_length],)

    def prediction_instance(self):
        return self._transform((self._X[:, -self._total_length :], self._interpolated_y[:, -self._context_length :]))


class SimpleDataLoader:
    def __init__(self, dataset: DataSet):
        self.dataset = dataset

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]


class DataLoader:
    def __init__(self, X, y, forecast_length, context_length=None, do_validation=False):
        self._X = X  # n_locations, n_periods, n_features
        self._y = y  # n_locations, n_periods
        self._interpolated_y = interpolate_nans(y)
        self._context_length = context_length or X.shape[1] - forecast_length
        self._forecast_length = forecast_length
        self._total_length = self._context_length + forecast_length
        self.validation_mask = np.ones(X.shape[1] - self._total_length + 1, dtype=bool)
        if do_validation:
            self._validation_index = (X.shape[1] - self._total_length) // 2
            self.validation_mask[
                self._validation_index - forecast_length : self._validation_index + forecast_length
            ] = False
        self.do_validation = do_validation

    def __len__(self):
        return np.sum(self.validation_mask)

    def __iter__(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        starts = np.arange(self._X.shape[1] - self._total_length + 1)[self.validation_mask]
        permuted_starts = np.random.permutation(starts) if False else starts
        return (
            (
                self._X[:, start : start + self._total_length],
                self._interpolated_y[:, start : start + self._context_length],
                self._y[:, start : start + self._total_length],
            )
            for start in permuted_starts
        )

    def validation_set(self):
        return (
            self._X[:, self._validation_index : self._validation_index + self._total_length],
            self._interpolated_y[:, self._validation_index : self._validation_index + self._context_length],
            self._y[:, self._validation_index : self._validation_index + self._total_length],
        )
