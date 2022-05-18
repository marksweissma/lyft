from typing import *
from numbers import Number
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from category_encoders import JamesSteinEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

import cloudpickle
from fire import Fire
import numpy as np
import pandas as pd
import pickle
import variants
import yaml

from training.utils import (Distance, Schema, KMeansHack,
                            TimeFeaturesProjectUnitCircle, PipelineXGBResample,
                            package_terminal_estimator_params)

DEFAULT_UPDATE_COUNT = 10000

TRANSFORMER_REGISTRY = {
    'distance': Distance,
    'schema': Schema,
    'kmeanshack': KMeansHack,
    'jamessteinencoder': JamesSteinEncoder,
    'onehotencoder': OneHotEncoder,
    'xgbregressor': XGBRegressor,
    'timefeaturesprojectunitcircle': TimeFeaturesProjectUnitCircle,
}


def write(obj: Any, location: str) -> None:
    with open(location, 'wb') as f:
        cloudpickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)


def query_range(df: pd.DataFrame, start: str, stop: str,
                field: Hashable) -> pd.DataFrame:
    start = pd.Timestamp(start)
    stop = pd.Timestamp(stop)
    return df.query(f'({field} >= @start) & ({field} < @stop)')


def load_data(file_path: str,
              splits: List[str],
              sample: Optional[int] = None,
              **kwargs) -> pd.DataFrame:
    print(f"loading data from {file_path}")
    converters = {
        # 'start_timestamp': lambda column: pd.to_datetime(column, unit='s'),
    }
    df = pd.read_csv(file_path, converters=converters, **kwargs)
    if sample:
        df = df.sample(sample)
    df.start_timestamp = pd.to_datetime(df.start_timestamp, unit='s')

    outputs = {
        key: query_range(df, start, stop, field)
        for key, (start, stop, field) in splits.items()
    }

    return outputs


def load_pipeline(pipeline_definition: List[Dict],
                  pipeline_kwargs: Dict = None,
                  sampler: Any = None) -> Pipeline:

    def _coalesce(value, enforced):
        return value if isinstance(value, enforced) else enforced()

    pipeline_kwargs = pipeline_kwargs if pipeline_kwargs else {}
    steps = [(key, TRANSFORMER_REGISTRY[key](**_coalesce(params, dict)))
             for key, params in pipeline_definition]
    if sampler:
        key, params = sampler
        sampler = TRANSFORMER_REGISTRY[key](**_coalesce(params, dict))
        pipeline_kwargs.update({'sampler': sampler})
    pipeline = PipelineXGBResample(steps=steps, **pipeline_kwargs)
    print(f"pipeline constructed: {pipeline}")
    return pipeline


@variants.primary
def train_model(variant: str = None,
                data: pd.DataFrame = None,
                pipeline: Pipeline = None,
                target: str = None,
                write_location: str = None,
                cutoff: Number = None,
                **fit_params) -> Pipeline:
    if cutoff:
        mask = data[target] < cutoff
        data = data.loc[mask]
    variant = variant if variant else 'base'
    pipeline = getattr(train_model, variant)(data, pipeline, target,
                                             **fit_params)
    if write_location:
        write(pipeline, f"models/{write_location}")
        write_location = '_'.join(['train', write_location])
        data_write_location = f"{write_location.split('.', 1)[0]}.csv"
        data.to_csv(f"data/{data_write_location}", index=False)
    return pipeline


@train_model.variant('base')
def train_model(data: pd.DataFrame, pipeline: Pipeline, target: str,
                **fit_params):
    print(f'fitting rows: {len(data)}')
    X = data.drop(target, axis=1)
    y = data[target]
    pipeline.fit(X, y.astype(int), **fit_params)

    return pipeline


@train_model.variant('xgb')
def train_model(data: pd.DataFrame,
                pipeline: Pipeline,
                target: str,
                xgb_fit_params: Dict = None,
                eval_split_kwargs: Dict = None,
                **fit_params) -> Pipeline:
    eval_split_kwargs = eval_split_kwargs if eval_split_kwargs else {}
    split_kwargs = eval_split_kwargs.get('split_kwargs', {})
    train, test = train_test_split(data, **split_kwargs)
    downsample = eval_split_kwargs.get('downsample', {})
    if downsample:
        test = test.sample(**downsample)

    if xgb_fit_params:
        xgb_fit_params['eval_set'] = [(test.drop(target,
                                                 axis=1), test[target])]
        fit_params.update(
            package_terminal_estimator_params(pipeline, xgb_fit_params))

    iterations = eval_split_kwargs.get('iterations', 1)
    iteration = 0
    sample, remainder = train_test_split(
        train, **eval_split_kwargs.get('remainder_kwargs', {}))
    while iteration < iterations:
        print(
            f'sample: {len(sample)}, remainder: {len(remainder)}, evaluation: {len(test)})'
        )

        pipeline = train_model(data=sample,
                               pipeline=pipeline,
                               target=target,
                               **fit_params)
        print(
            f'{iteration}: {calculate_performance_metrics(test[target], pipeline.predict(test))}'
        )
        scores = pd.Series(remainder[target] -
                           pipeline.predict(remainder.drop(target, axis=1)),
                           index=remainder.index).abs().sort_values()
        update_count = eval_split_kwargs.get('update_count',
                                             DEFAULT_UPDATE_COUNT)
        indices = scores.sort_values().iloc[-update_count:].index
        sample = pd.concat([sample, remainder.loc[indices]])
        remainder = remainder.drop(indices)
        assert len(sample) + len(remainder) == len(train)
        assert len(sample.index.intersection(remainder.index)) == 0
        iteration += 1

    return pipeline


def calculate_performance_metrics(labels: Iterable,
                                  predictions: Iterable) -> Dict[str, Number]:
    return {
        'count':
        len(labels),
        'r2 score':
        r2_score(labels, predictions),
        'mean_absolute_error':
        mean_absolute_error(labels, predictions),
        'mean_absolute_percentage_error':
        mean_absolute_percentage_error(labels, predictions),
        'mean_squared_error':
        mean_squared_error(labels, predictions),
    }


def evaluate(data: pd.DataFrame,
             pipeline: Pipeline,
             write_location: str,
             target: str = None,
             prefix: str = None,
             **kwargs):
    predictions = pd.Series(pipeline.predict(data),
                            name='prediction',
                            index=data.index)
    data = pd.concat([data, predictions], axis=1)
    performance = calculate_performance_metrics(
        data[target], pd.Series(predictions.clip(lower=1)))
    print(performance)

    data_write_location = f"{write_location.split('.', 1)[0]}.csv"
    data.to_csv(f"data/{data_write_location}", index=False)


def generate_result(model: BaseEstimator, read_location: str,
                    write_location: str):
    converters = {
        'start_timestamp': lambda column: pd.to_datetime(column, unit='s'),
    }
    df = pd.read_csv(read_location, converters=converters)
    predictions = model.predict(df)

    # variable name from assignment sheet
    pred_duration = pd.DataFrame(predictions,
                                 columns=['duration'],
                                 index=df.row_id)
    pred_duration.duration = pred_duration.duration.clip(lower=1)
    pred_duration.to_csv(write_location)


def main(conf: Dict):
    with open(conf, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    data = load_data(**conf.get('data', {}))
    pipeline = load_pipeline(**conf.get('estimator', {}))
    pipeline = train_model(data=data['train'],
                           pipeline=pipeline,
                           **conf.get('training', {}))
    evaluate(data['test'],
             pipeline,
             prefix='test',
             **conf.get('evaluation', {}))

    generate_result(pipeline, **conf.get('result', {}))


if __name__ == '__main__':
    Fire(main)
