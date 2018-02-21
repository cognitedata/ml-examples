import pandas as pd

def generate_supervised_dataset_w_lookback(df, lookback):
    columns = []
    for i in range(1, lookback + 1):
        columns.append(df.shift(i).iloc[:,:-1])
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

def preprocess(df):
    df = df.iloc[:, 1:]
    X, y = generate_supervised_dataset_w_lookback(df, 1)
    X, y = reshape_data(X, y)
    return X, y
