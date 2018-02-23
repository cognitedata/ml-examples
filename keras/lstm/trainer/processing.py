'''Data processing module required for compatibility with Cognite Model Hosting Environment'''
import os

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


def preprocess(df, pp_dir):
    '''Preprocesses data for training.

    Saves necessary preprocessing objects to designated directory; --pp-dir or a local directory, so that the same
    preprocessing can be performed on prediction data.
    '''
    df = df.iloc[:, 1:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = pd.DataFrame(scaler.fit_transform(df.values))

    if pp_dir.startswith("gs://"):
        joblib.dump(scaler, 'scaler.save')
        copy_file_to_gcs(pp_dir, 'scaler.save')
    else:
        joblib.dump(scaler, os.path.join(pp_dir, 'scaler.save'))

    X, y = generate_supervised_dataset_w_lookback(df, 1)
    X, y = reshape_data(X, y)
    return X, y


def preprocess_prediction_data(df, pp_dir):
    '''Preprocesses data for prediction.

    Preprocesses a dataframe of prediction data in the same way the training data was processed. To do this it loads
    the scaler saved in the preprocess method.

    Note:
        This method is required for compatibility with the Cognite Model Hosting Environment.
    '''

    # Do prepocessing
    df = df.iloc[:, 1:]

    # Load
    scaler = joblib.load(os.path.join(pp_dir, 'scaler.save'))

    df = pd.DataFrame(scaler.transform(df.values))
    X, y = generate_supervised_dataset_w_lookback(df, 1)
    X, y = reshape_data(X, y)
    return X, y


def inverse_transform_prediction(prediction, pp_dir):
    '''Inverts scaling for a single prediction result.

    Does not alter shape of prediction result.

    Note:
        This method is required for compatibility with the Cognite Model Hosting Environment.
    '''
    scaler = joblib.load(os.path.join(pp_dir, 'scaler.save'))
    input = scaler.data_range_
    input[-1] = prediction[0]
    return [scaler.inverse_transform([input])[0][-1]]


def copy_file_to_gcs(dir, file_path):
    '''Copy a file to Google Cloud Storage.'''
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    df = pd.DataFrame([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])

    # preprocess(df, './')
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        pp_dir_contents = 'gs://cognite-ml/jobs/forecasting/001256b3a1ac7d347c88376b7f3b4f2a90a/pp/*'
        temp_pp_dir = tmp_dir + '/pp'
        # Unhappy hack to workaround gcs client library not being able to support downloading entire contents of
        # bucket folder
        os.mkdir(temp_pp_dir)
        os.system('gsutil cp {} {}'.format(pp_dir_contents, temp_pp_dir))

        prediction = [0.598]

        print(inverse_transform_prediction(prediction, temp_pp_dir))
