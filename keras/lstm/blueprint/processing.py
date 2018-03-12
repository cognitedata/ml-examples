import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.lib.io import file_io


def generate_supervised_dataset_w_lookback(df, lookback):
    columns = []
    for i in range(1, lookback + 1):
        columns.append(df.shift(i).iloc[:, :-1])
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df = df.dropna()
    X, y = df.values[:, :-1], df.values[:, -1].reshape(-1, 1)
    return X, y


def reshape_data(X, y):
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    X = X.reshape(n_rows, 2, int(n_cols / 2))
    y = y.reshape(-1, 1)
    return X, y


def preprocess(df, pp_dir=None):
    '''Preprocesses data for training.

    Saves necessary preprocessing objects to designated directory; --pp-dir or a local directory, so that the same
    preprocessing can be performed on prediction data.
    '''
    df = df.iloc[:, 1:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = pd.DataFrame(scaler.fit_transform(df.values))

    if pp_dir is not None and pp_dir.startswith("gs://"):
        joblib.dump(scaler, 'scaler.save')
        copy_file_to_gcs(pp_dir, 'scaler.save')
    else:
        joblib.dump(scaler, 'scaler.save')

    X, y = generate_supervised_dataset_w_lookback(df, 1)
    X, y = reshape_data(X, y)
    return X, y


def copy_file_to_gcs(dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def preprocess_prediction_data(df):
    '''Preprocesses data for prediction.

    Preprocesses a dataframe of prediction data in the same way the training data was processed. To do this it loads
    the scaler saved in the preprocess method.
    '''
    df = df.iloc[:, 1:]
    current_dir = os.path.dirname(__file__)
    scaler = joblib.load(current_dir + '/scaler.save')

    df = pd.DataFrame(scaler.transform(df.values))
    X, y = generate_supervised_dataset_w_lookback(df, 1)
    X, _ = reshape_data(X, y)
    return X


def inverse_transform_prediction(results):
    '''Inverts scaling for prediction results.

    Does not alter shape of prediction result.
    '''
    current_dir = os.path.dirname(__file__)
    scaler = joblib.load(current_dir + '/scaler.save')

    transformed_results = []
    for prediction in results:
        input = scaler.data_range_
        input[-1] = prediction[0]
        transformed_results.append([scaler.inverse_transform([input])[0][-1]])
    return np.array(transformed_results)
