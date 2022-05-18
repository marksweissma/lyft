from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, _final_estimator_has, _name_estimators
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.cluster import KMeans
from attrs import define, field, Factory
from functools import singledispatch
from itertools import chain
from pandas.tseries.holiday import USFederalHolidayCalendar

import pandas as pd
import numpy as np

from typing import *

HOLIDAYS = USFederalHolidayCalendar().holidays(
    start=pd.Timestamp('2015-01-01'), end=pd.Timestamp('2015-12-31'))

SECONDS_IN_DAY = 24 * 3600
SECONDS_IN_WEEK = 7 * 24 * 3600


@define
class Distance(TransformerMixin, BaseEstimator):
    lat_kwargs: Dict = Factory(dict)
    lng_kwargs: Dict = Factory(dict)
    rect_kwargs: Dict = Factory(dict)
    lat_cut: List = Factory(list)
    lng_cut: List = Factory(list)
    rect_cut: Dict = Factory(dict)

    @staticmethod
    def generate_cut(series, lower, upper, bin_count=11):
        percentiles = np.linspace(0, 1, bin_count)
        labels = series.describe(percentiles=percentiles).drop(
            ['count', 'mean', 'std', 'min', 'max']).tolist()
        return [lower] + labels + [upper]

    def fit(self, X, y=None, **fit_params):
        locations = pd.concat([
            X[['start_lat', 'start_lng']].rename(columns={
                'start_lat': 'lat',
                'start_lng': 'lng'
            }), X[['end_lat', 'end_lng']].rename(columns={
                'end_lat': 'lat',
                'end_lng': 'lng'
            })
        ])
        self.lat_cut = self.generate_cut(locations.lat, -90, 90,
                                         **self.lat_kwargs)
        self.lng_cut = self.generate_cut(locations.lng, -180, 180,
                                         **self.lng_kwargs)

        rect_lat_cut = self.generate_cut(locations.lat, -90, 90,
                                         **self.rect_kwargs.get('lat', {}))
        rect_lng_cut = self.generate_cut(locations.lng, -180, 180,
                                         **self.rect_kwargs.get('lng', {}))
        self.rect_cut = {'lat': rect_lat_cut, 'lng': rect_lng_cut}
        return self

    def transform(self, X):
        X['lat_delta'] = X['end_lat'] - X['start_lat']
        X['lng_delta'] = X['end_lng'] - X['start_lng']

        lat_start_cut = pd.cut(X.start_lat,
                               bins=self.rect_cut['lat'],
                               labels=False).astype(str)
        lng_start_cut = pd.cut(X.start_lng,
                               bins=self.rect_cut['lng'],
                               labels=False).astype(str)
        lat_end_cut = pd.cut(X.end_lat,
                             bins=self.rect_cut['lat'],
                             labels=False).astype(str)
        lng_end_cut = pd.cut(X.end_lng,
                             bins=self.rect_cut['lng'],
                             labels=False).astype(str)
        X['start_cut'] = lat_start_cut + lng_start_cut
        X['end_cut'] = lat_end_cut + lng_end_cut

        return X


def as_factory(handler):
    """
    converter for handler to accept object or a `dict` config
    for a handler to generate an object. handlers are callable
    (functions or types).
    """

    @singledispatch
    def _as(obj):
        return obj

    @_as.register(dict)
    def _as_from_conf(conf):
        obj = handler(**conf)
        return obj

    return _as


@singledispatch
def infer_schema(obj):
    raise TypeError(f'obj: {obj} type: {type(obj)} not supported')


@infer_schema.register(pd.DataFrame)
def infer_schema_df(obj):
    schema = {}
    [schema.update(infer_schema(obj[column])) for column in obj]
    return schema


@infer_schema.register(pd.Series)
def infer_schema_series(obj):
    first_valid_value = obj.loc[obj.first_valid_index()]
    return {obj.name: type(first_valid_value)}


@define
class Select:
    include: Optional[List] = None
    exclude: Optional[List] = None

    def __call__(self, df):
        include = self.include
        exclude = self.exclude
        keys = [i for i in df
                if i in include] if include is not None else list(df)
        if exclude:
            keys = [i for i in keys if i not in exclude]
        return df[keys]


@define
class Schema(TransformerMixin, BaseEstimator):
    selection: Callable = field(converter=as_factory(Select),
                                default=Factory(dict))
    infer_schema: Callable = infer_schema
    schema: Dict = Factory(dict)

    def build_schema(self, **kwargs):
        for key, value in kwargs.items():
            self.schema[key] = self.infer_schema(value)
        return self.schema

    def fit(self, X, y=None, **fit_params):
        self.schema = {}
        X_t = self.selection(X)
        payload = {'X': X_t}
        payload.update({'y': y}) if y is not None else None

        self.build_schema(**payload)
        return self

    def transform(self, X):
        return X[self.schema['X']]


class KMeansHack(KMeans):

    def __init__(self,
                 n_clusters=8,
                 *,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 tol=0.0001,
                 verbose=0,
                 random_state=None,
                 copy_x=True,
                 algorithm='auto'):
        super().__init__(n_clusters,
                         init=init,
                         n_init=n_init,
                         max_iter=max_iter,
                         tol=tol,
                         verbose=verbose,
                         random_state=random_state,
                         copy_x=copy_x,
                         algorithm=algorithm)

    def fit(self, X, y=None, **fit_params):
        start = X[['start_lat', 'start_lng']].values
        end = X[['end_lat', 'end_lng']].values
        return super().fit(np.vstack([start, end]), **fit_params)

    def _transform_one(self, X, name):
        index = X.index
        labels = pd.Series(super().predict(X.values),
                           name=f"{name}_cluster",
                           index=index).astype(str)
        distances = pd.DataFrame(super().transform(X.values),
                                 index=index).add_prefix(name)
        return [labels, distances]

    def transform(self, X):
        start = self._transform_one(X[['start_lat', 'start_lng']], 'start')
        end = self._transform_one(X[['end_lat', 'end_lng']], 'end')
        return pd.concat([X] + start + end, axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=y, **fit_params)
        return self.transform(X)


@define
class TimeFeaturesProjectUnitCircle(TransformerMixin, BaseEstimator):
    columns: List[Hashable]
    deltas: List[Hashable] = field()

    @deltas.default
    def default_deltas(self) -> List[Hashable]:
        return list(itertools.combinations(self.columns, 2))

    delta_unit: str = '1D'

    def single_column_features(self, df: pd.DataFrame,
                               column: str) -> pd.DataFrame:
        second_of_day = df[column].dt.hour * 3600 + df[
            column].dt.minute * 60 + df[column].dt.second
        second_of_week = df[
            column].dt.dayofweek * SECONDS_IN_DAY + second_of_day
        df[f'{column}_time_of_day_x'] = np.cos(2 * np.pi * second_of_day /
                                               SECONDS_IN_DAY)
        df[f'{column}_time_of_day_y'] = np.sin(2 * np.pi * second_of_day /
                                               SECONDS_IN_DAY)
        df[f'{column}_time_of_week_x'] = np.cos(2 * np.pi * second_of_week /
                                                SECONDS_IN_WEEK)
        df[f'{column}_time_of_week_y'] = np.sin(2 * np.pi * second_of_week /
                                                SECONDS_IN_WEEK)
        df['is_weekend'] = df[column].dt.dayofweek > 4
        df['is_holiday'] = df[column].dt.date.astype('datetime64').isin(
            HOLIDAYS)

        return df

    def fit(self, X, y=None, **fit_params) -> TimeFeaturesProjectUnitCircle:
        return self

    def transform(self, X) -> pd.DataFrame:
        for column in self.columns:
            X = self.single_column_features(X, column)

        for start, stop in self.deltas:
            X[f'{start}__{stop}_delta'] = (X[stop] - X[start]) / pd.Timedelta(
                self.delta_unit)
        columns = set(self.columns).union(chain.from_iterable(self.deltas))
        return X.drop(columns, axis=1)


def package_terminal_estimator_params(
    pipeline: Pipeline,
    params: Dict,
    instance_check: Callable = lambda x: hasattr(x, 'steps')
) -> Dict:
    name, transform = pipeline.steps[-1]
    if instance_check(transform):
        packaged_params = package_terminal_estimator_params(
            transform, params, instance_check)
        updated_params = {
            f'{name}__{key}': value
            for key, value in packaged_params.items()
        }
    else:
        updated_params = {
            f'{name}__{key}': value
            for key, value in params.items()
        }
    return updated_params


def transform_xgb_eval_set_if_in_fit_params(pipeline: Pipeline,
                                            **fit_params: Dict) -> Dict:
    if 'eval_set' in fit_params:
        transformed_eval_sets = []
        for X, y in fit_params['eval_set']:
            Xt = X  # sklearn convention
            for _, name, transform in pipeline._iter(with_final=False):
                Xt = transform.transform(Xt)
            transformed_eval_sets.append((Xt, y))
        fit_params['eval_set'] = transformed_eval_sets
    return fit_params


class PipelineXGBResample(Pipeline):

    def __init__(self, steps, memory=None, verbose=False, sampler=None):
        if sampler is not None and not hasattr(sampler, 'fit_resample'):
            raise TypeError(f'sampler: {sampler} is not a valid resampler')
        self.sampler = sampler
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params) -> PipelineXGBResample:
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        if self.sampler is not None:
            Xt, y = self.sampler.fit_resample(Xt, y)

        with _print_elapsed_time("Pipeline",
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                fit_params_last_step = transform_xgb_eval_set_if_in_fit_params(
                    self, **fit_params_last_step)
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        if sampler is not None:
            Xt, y = self.sampler.fit_resample(Xt, y)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline",
                                 self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            fit_params_last_step = transform_xgb_eval_set_if_in_fit_params(
                self, fit_params_last_step)
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, y, **fit_params_last_step)
            else:
                return last_step.fit(Xt, y,
                                     **fit_params_last_step).transform(Xt)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        if sampler is not None:
            Xt, y = self.sampler.fit_resample(Xt, y)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline",
                                 self._log_message(len(self.steps) - 1)):
            fit_params_last_step = transform_xgb_eval_set_if_in_fit_params(
                self, fit_params_last_step)
            y_pred = self.steps[-1][1].fit_predict(Xt, y,
                                                   **fit_params_last_step)
        return y_pred


def make_pipeline_xgb_resample(*steps,
                               memory=None,
                               verbose=False,
                               sampler=None) -> PipelineXGBResample:
    return PipelineXGBResample(_name_estimators(steps),
                               memory=memory,
                               verbose=verbose,
                               sampler=sampler)
