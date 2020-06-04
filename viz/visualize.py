# Title     : visualization module
# Objective : framework for generalized visualization
# Created by: abhi
# Created on: 6/4/20

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings('ignore')


def density_plot(data_frame, column):
    sns.distplot(data_frame[column])
    plt.show()
    # skewness and kurtosis
    print("Skewness: %f" % data_frame[column].skew())
    print("Kurtosis: %f" % data_frame[column].kurt())


def scatter_plot(data_frame, dependent, column):
    data = pd.concat([data_frame[dependent], data_frame[column]], axis=1)
    data.plot.scatter(x=column, y=dependent, ylim=(0, 800000))
    plt.show()


def box_plot(data_frame, dependent, column):
    data = pd.concat([data_frame[dependent], data_frame[column]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=column, y=dependent, data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()


def corr_mat(data_frame):
    corrmat = data_frame.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.show()


def corr_mat_topk(data_frame, dependent, k):
    corrmat = data_frame.corr()
    cols = corrmat.nlargest(k, dependent)[dependent].index
    cm = np.corrcoef(data_frame[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


def pair_plot(data_frame, columns):
    sns.set()
    sns.pairplot(data_frame[columns], size=2.5)
    plt.show()


def hist_normal_prob(data_frame, dependent):
    sns.distplot(data_frame[dependent], fit=norm);
    fig = plt.figure()
    res = stats.probplot(data_frame[dependent], plot=plt)
    plt.show()


def missing_data_plot(data_frame):
    data_frame_na = (data_frame.isnull().sum() / len(data_frame)) * 100
    data_frame_na = data_frame_na.drop(data_frame_na[data_frame_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio': data_frame_na})
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=data_frame_na.index, y=data_frame_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv("../input/train_house_price.csv")
    # density_plot(df, 'SalePrice')
    # scatter_plot(df, 'SalePrice', 'GrLivArea')
    # box_plot(df, 'SalePrice', 'OverallQual')
    # corr_mat(df)
    # corr_mat_topk(df, 'SalePrice', 10)
    # pair_plot(df, ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'])
    #hist_normal_prob(df, 'SalePrice')
    missing_data_plot(df)
