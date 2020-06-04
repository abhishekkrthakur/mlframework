import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings('ignore')


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def gen_eda(data_frame):

    print("##########----------##########")
    print(f'Dataset has {data_frame.shape[0]} rows and {data_frame.shape[1]} columns.')

    print("##########----------##########")
    print(f'There are {data_frame.isnull().any().sum()} columns in the dataset with missing values.')

    print("##########----------##########")
    one_value_cols = [col for col in data_frame.columns if data_frame[col].nunique() <= 1]
    print(f'There are {len(one_value_cols)} columns in the dataset with one unique value.')

    print("##########----------##########")
    dtype_df = data_frame.dtypes.reset_index()
    dtype_df.columns = ["Count", "Column Type"]
    print(dtype_df)

    print("##########----------##########")
    df1 = dtype_df.groupby("Column Type").aggregate('count').reset_index()
    print(df1)

    print("##########----------##########")
    # Number of unique classes in each object column
    df2 = data_frame.select_dtypes('object').apply(pd.Series.nunique, axis=0)
    print(df2)


def general_stats(data_frame):
    stats = []
    for col in data_frame.columns:
        stats.append((col, data_frame[col].nunique(), data_frame[col].isnull().sum() * 100 / data_frame.shape[0],
                      data_frame[col].value_counts(normalize=True, dropna=False).values[0] * 100, data_frame[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Percentage of missing values', ascending=False)
    print(stats_df)

def find_correlations(data_frame, dependent):
    # Find correlations with the target and sort
    correlations = data_frame.corr()[dependent].sort_values()

    # Display correlations
    print('Most Positive Correlations:\n', correlations.tail(15))
    print('\nMost Negative Correlations:\n', correlations.head(15))


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def print_quantiles(data_frame, column):
    data_frame[column] = data_frame[column].astype(float)
    print(f"{column} Quantiles:")
    print(data_frame[column].quantile([.01, .025, .1, .25, .5, .75, .9, .975, .99]))


if __name__ == "__main__":
    df = pd.read_csv("../input/train_house_price.csv")
    #gen_eda(df)
    #general_stats(df)
    #find_correlations(df, 'SalePrice')
    missing_values_table(df)
