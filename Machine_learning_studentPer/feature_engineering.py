#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import lightgbm
from lightgbm.sklearn import LGBMRegressor
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm

# Load raw data
train = pd.read_csv("train.csv")


def summary(df):
    '''
    Create a table to describe data
    Args:
        df: raw data
    Returns:
        summ: data distribution
    '''
    summ = train.describe(include='all').transpose()
    summ = summ.drop(['top', 'freq', '25%', '75%'], axis=1)
    rows = len(df)
    summ['#missing'] = summ['count'].apply(lambda x: rows - x)
    summ['%missing'] = summ['#missing'].apply(lambda x: x / rows * 100)
    #     summ = summ.style.format("{:,.2f}")
    values = []
    for col in df:
        value = df[col].value_counts().to_dict()
        if len(value) < 130:
            values.append(value)
        else:
            values.append('')
    summ['#unique_values'] = values

    return summ


# Describe data
summary_table = summary(train)
print(summary_table)

# Fill empty values
train.text_fqid.fillna(train.room_fqid + '.' + train.fqid, inplace=True)
# Delete redundant, missing features
train = train.drop(['index', 'page', 'level', 'name', 'text', 'fqid'], axis=1)


def feature_engineering(train):
    '''
    Process raw feature data
    Args:
        train: raw data
    Returns:
        df: processed feature data
    '''
    cats = ['text_fqid', 'room_fqid']
    coors = ['room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y']
    time = ['elapsed_time', 'hover_duration']
    modes = ['fullscreen', 'hq', 'music']
    events = train['event_name'].unique().tolist()

    # Dummy coding to describe qualitative data quantitatively
    dummies = pd.get_dummies(train['event_name'])
    event_table = pd.concat([train[['session_id', 'event_name', 'level_group']], dummies], axis=1)

    dfs = []

    # get unique value for categories
    for c in cats:
        tmp = train.groupby(['session_id', 'level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)

    # get mean and standard deviation value for coordinates, time and modes
    distribution = coors + time + modes
    for d in distribution:
        tmp = train.groupby(['session_id', 'level_group'])[d].agg(['mean', 'std'])
        tmp = tmp.rename({'mean': d + '_mean', 'std': d + '_std'}, axis=1)
        dfs.append(tmp)

    # get sum value for unique events
    for e in events:
        tmp = event_table.groupby(['session_id', 'level_group'])[e].agg('sum')
        tmp.name = tmp.name + '_sum'
        dfs.append(tmp)

    # get sum value for time
    for t in time:
        tmp = train.groupby(['session_id', 'level_group'])[t].agg('sum')
        tmp.name = tmp.name + '_sum'
        dfs.append(tmp)

    # collect all processed data and fill nan
    df = pd.concat(dfs, axis=1)
    df = df.fillna(0)

    # reset index with session_id
    df = df.reset_index()
    df = df.set_index('session_id')

    return df


# Apply feature engineering
df = feature_engineering(train)
print(df.info())

# Save processed data
df.to_csv('feature_data.csv')

# Apply feature engineering
test = pd.read_csv("test.csv")
df_test = feature_engineering(test)
print(df_test.info())

# Save processed data
df_test.to_csv('test_feature_data.csv')
