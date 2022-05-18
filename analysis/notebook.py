from matplotlib import pyplot as plt
from functools import lru_cache
from wrapt import decorator

import numpy as np
import pandas as pd
import seaborn as sns
import cloudpickle
import shap
import xgboost
import variants
from typing import *

pd.options.mode.chained_assignment = None

RAW_DATA = './data/train.csv'


@decorator
def copy(wrapped, instance, args, kwargs):
    return wrapped(*args, **kwargs).copy()


def load_data(file_path: str = RAW_DATA):
    df = pd.read_csv(file_path)
    return df


@copy
@lru_cache()
def load_sample(sample):
    df = load_data()
    sample = df.sample(sample)
    sample.start_timestamp = pd.to_datetime(sample.start_timestamp, unit='s')
    sample['date'] = sample.start_timestamp.dt.date
    return sample


@decorator
def default_data(wrapped, instance, args, kwargs):
    if len(args) < 1 and 'df' not in kwargs:
        kwargs['df'] = load_data()
    return wrapped(*args, **kwargs)


@decorator
def default_sample(wrapped, instance, args, kwargs):
    sample = args[0] if args else kwargs['sample']
    if isinstance(sample, int):
        sample = load_sample(sample)
    if args:
        args = (sample, *args[1:])
    else:
        kwargs['sample'] = sample

    return wrapped(*args, **kwargs)


@default_data
def summary(df=None):
    print(df.shape)
    return pd.concat([
        df.isnull().mean().rename('is_null_fraction'),
        df.dtypes.rename('schema'),
        df.dropna().head(5).T
    ],
                     axis=1)


@default_sample
def scatter(sample=200000,
            x='lng',
            y='lat',
            prefix=None,
            c=None,
            cmap=None,
            clip=None,
            **kwargs):
    if clip:
        c = sample[c].clip(**clip)
    if prefix:
        x = f'{prefix}_{x}'
        y = f'{prefix}_{y}'
    sample.plot(kind='scatter', x=x, y=y, c=c, cmap=cmap, **kwargs)


try:
    render = display
except NameError:
    render = print


@default_data
def duration_summary(df, scale=60, name='minutes', sample=200000):

    render(f'number records: {len(df)}')
    render((df.duration / scale).describe(
        percentiles=[.25, .5, .75, .9, .95, .997, .9999]).rename(
            'duration - minutes').to_frame().drop('count'))
    render(f'sample size for histogram: {sample}')
    sns.displot(df.sample(sample).duration.rename('duration_minutes') / 60,
                bins=range(0, 100))


@default_sample
def rides_by_day(sample=200000):
    sample['date'] = sample.start_timestamp.dt.date
    sample.groupby('date').row_id.count().rename('ride volume').plot(
        figsize=(12, 8))


@default_sample
def duration_by_day(sample=200000):
    (sample.groupby('date').duration.describe()[['25%', 'mean', '75%']] /
     60).plot(figsize=(12, 8), title='duration_minutes')


def _load(location):
    with open(location, 'rb') as f:
        return cloudpickle.load(f)


def load_explaination(model='models/model_xgb.pkl',
                      data='data/train_model_xgb.csv',
                      sample=5000):
    estimator = _load(model)
    data = pd.read_csv(data,
                       converters={'start_timestamp': pd.to_datetime},
                       nrows=sample)

    for _, name, transform in estimator._iter(with_final=False):
        data = transform.transform(data)
    explainer = shap.TreeExplainer(estimator[-1])
    shap_values = explainer.shap_values(data)
    xgboost.plot_importance(estimator[-1],
                            importance_type="gain",
                            max_num_features=15)

    plt.show()
    shap.summary_plot(shap_values, data)


@lru_cache()
def load_evaluation(data='data/evaluation_xgb.csv', nrows=200000):
    evaluation = pd.read_csv(data,
                             converters={'start_timestamp': pd.to_datetime},
                             nrows=nrows)

    evaluation['date'] = evaluation.start_timestamp.dt.date
    evaluation['residual'] = evaluation.duration - evaluation.prediction
    evaluation['residual_abs'] = evaluation.residual.abs()
    evaluation[
        'residual_precent'] = evaluation.residual_abs / evaluation.duration * 100

    return evaluation


def residual_hist():
    render('residuals (seconds)')
    evaluation = load_evaluation()
    evaluation.residual.hist(
        bins=range(-1600, 1600, 15),
        figsize=(8, 6),
    )


def residual_scatter():
    render('residuals (seconds)')
    evaluation = load_evaluation()
    scatter(evaluation.sort_values('residual_abs').tail(5000),
            'lng',
            'lat',
            'start',
            'residual',
            'viridis',
            clip={
                'upper': 1500,
                'lower': -1200
            },
            figsize=(18, 12),
            alpha=.25)


def residual_by_day():
    render('abs(residuals) (seconds)')
    evaluation = load_evaluation()
    evaluation.groupby('date').residual_abs.describe(
        percentiles=[.25, .75, .9, .99])[['25%', 'mean', '75%', '90%',
                                          '99%']].plot(figsize=(12, 8))
