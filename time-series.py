import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def data_uploading():
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    return df[5::6]


def show_trend(df):
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)
    plt.show()

    # plot_features = df[plot_cols][:480]
    # plot_features.index = date_time[:480]
    # _ = plot_features.plot(subplots=True)


def remove_error_from_wind_velocity(df):
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame.
    # df['wv (m/s)'].min()


def generate_wind_variables(df):
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)


def time_signal(df):
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    return df


def train_test_split(df):
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]

    return train_df, val_df, test_df, num_features, column_indices


def normalize(train_df, val_df, test_df):
    df_mean = train_df.mean()
    df_std = train_df.std()
    return df_std, (train_df - df_mean) / df_std, (val_df - df_mean) / df_std, (test_df - df_mean) / df_std


def main():
    df = data_uploading()
    # show_trend(df)
    remove_error_from_wind_velocity(df)
    generate_wind_variables(df)
    train_df, val_df, test_df, num_features, column_indices = train_test_split(df)
    train_std, train_df, val_df, test_df = normalize(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
