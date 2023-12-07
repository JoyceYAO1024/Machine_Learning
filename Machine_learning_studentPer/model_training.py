#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import lightgbm
from lightgbm.sklearn import LGBMRegressor
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm


# Load labels data
train_labels = pd.read_csv("train_labels.csv")

# Split session id and question id
train_labels['session'] = train_labels['session_id'].apply(lambda x:int(x.split('_')[0]))
train_labels['q'] = train_labels['session_id'].apply(lambda x:int(x.split('_')[-1][1:]))
train_labels = train_labels.drop('session_id', axis=1)
train_labels = train_labels.rename({'session':'session_id'}, axis=1)
train_labels = train_labels.set_index('session_id')

# Load features data
df = pd.read_csv("feature_data.csv", index_col='session_id')


def get_data_q(q, df, train_labels):
    '''
    Get Data according to question id
    Args:
        q: question id
        df: feature data
        train_labels: label data
    Returns:
        train_x: feature data of question id
        train_y: label data of question id
    '''
    if q <= 4:
        l = '0-4'
    elif q <= 12:
        l = '5-12'
    elif q <= 22:
        l = '13-22'
    train_x = df[df['level_group']==l].drop('level_group',axis=1)
    train_y = train_labels[train_labels['q']==q]
    
    return (train_x, train_y)


def feature_selecting(df, train_labels):
    '''
    Select features using GradientBoosting
    Args:
        df: all feature data
        train_labels: label data
    Returns:
        features: list of selected features
    '''
    features = []
    for q in tqdm(range(1,19)):
        train_x, train_y = get_data_q(q, df, train_labels)
        train_y = train_y['correct']
        model = SelectFromModel(GradientBoostingClassifier())
        model.fit_transform(train_x, train_y)
        features += model.get_feature_names_out().tolist()
    return list(set(features))


# Apply feature selecting
features = feature_selecting(df, train_labels)
print(f'Train {len(features)} features:')
print(features)
df_selected = df[['level_group']+features]


# Get # of session
users = df.index.unique()
print(f'Train {len(users)} sessions info.')


def nn_classifier(X_train, y_train, hidden_layer_sizes=(50, 100, 50), alpha=0.05, solver='adam', learning_rate='adaptive'):
    """
    Creates and trains a neural network classifier on the provided training data.
    
    Parameters:
    hidden_layer_sizes (tuple): Tuple specifying the number of neurons in each hidden layer (default is (100,))
    activation (str): Activation function to use for the hidden layers (default is 'relu')
    solver (str): Solver algorithm to use (default is 'adam')
    
    Returns:
    nn_model (Neural network model object): Trained neural network model
    """
    
    # Create neural network model object with specified parameters
    nn_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, solver=solver, learning_rate=learning_rate)
    
    # Fit the model on the training data
    nn_model.fit(X_train, y_train)
    
    return nn_model


def svm_classifier(X_train, y_train, kernel='rbf'):
    """
    Creates and trains an SVM classifier on the provided training data.
    
    Parameters:
    kernel (string): SVM kernel function (default is linear)
    
    Returns:
    svm_model (SVM model object): Trained SVM model
    """
    
    # Create SVM model object with specified kernel and regularization parameter
    svm_model = svm.SVR(kernel=kernel)
    
    # Fit the model on the training data
    svm_model.fit(X_train, y_train)
    
    return svm_model


def knn_classifier(X_train, y_train, n_neighbors=5):
    """
    Creates and trains a KNN classifier on the provided training data.
    
    Parameters:
    n_neighbors (int): Number of neighbors to use for classification (default is 5)
    
    Returns:
    knn_model (KNN model object): Trained KNN model
    """
    
    # Create KNN model object with specified number of neighbors
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    # Fit the model on the training data
    knn_model.fit(X_train, y_train)
    
    return knn_model


def adaboost_classifier(X_train, y_train, n_estimators=100):
    
    # Create AdaBoost model object with specified parameters
    adaboost_model = AdaBoostClassifier(n_estimators=n_estimators)
    
    # Fit the model on the training data
    adaboost_model.fit(X_train, y_train)
    
    return adaboost_model


def lgbm_classifier(X_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.01):

    # Create lightGBM model object with specified parameters
    lgbm_model = LGBMRegressor(learning_rate=learning_rate,num_leaves=15,n_estimators=n_estimators,
                               min_child_samples=20,boosting_type='gbdt',subsample_for_bin=1000,
                               max_depth=max_depth,colsample_bytree=0.8)
    
    # Fit the model on the training data
    lgbm_model.fit(X_train, y_train)
    
    return lgbm_model


def random_forest_classifier(X_train, y_train, n_estimators=300, max_depth=5):

    # Create random forest model object with specified parameters
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth) 
    
    # Fit the model on the training data
    rf_model.fit(X_train, y_train)
    
    return rf_model


def xgb_classifier(X_train, y_train, learning_rate=0.01, max_depth=3, n_estimators=100):

    # Create XGBoost model object with specified parameters
    xgb_params = {
    'objective' : 'binary:logistic',
    'eval_metric':'logloss',
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'n_estimators': n_estimators,
    'early_stopping_rounds': 50,
    'tree_method':'hist',
    'subsample':0.8,
    'colsample_bytree': 0.4,
    'use_label_encoder' : False}
    xgb_model = XGBClassifier(**xgb_params)
    
    # Fit the model on the training data
    xgb_model.fit(X_train, y_train)
    
    return xgb_model


def model_training(model_name, df, train_labels):
    '''
    Fit data to different models
    Args:
        model_name: used model
        df: feature data
        train_labels: label data
    Returns:
        models: trained model
        oof: predict probability
    '''
    n_folds = 5
    # use GroupKFold to cross-validation
    gkf = GroupKFold(n_splits = n_folds)

    # create an empty dataframe to store predict value
    oof = pd.DataFrame(data=np.empty((len(users),18),object),index=users)

    models = {}

    for i, (train_index, test_index) in enumerate(gkf.split(X=df, groups=df.index)):
        print(f"Fold {i}:")
        for q in tqdm(range(1,19)):
            X, y = get_data_q(q, df, train_labels)
            # train data
            train_users = df.index[train_index].unique()
            train_x = X.loc[train_users]
            train_y = y.loc[train_users]

            # valid data
            valid_users = df.index[test_index].unique()
            valid_x = X.loc[valid_users]
            valid_y = y.loc[valid_users]

            # save model, predict valid off
            if model_name in ['xgb', 'adaboost', 'random_forest']:
                clf = eval(model_name+'_classifier')(train_x,train_y['correct'])
                models[f'{i}_{q}'] = clf
                oof.loc[valid_users, q-1] = clf.predict_proba(valid_x)[:,1]
            else:
                model = eval(model_name+'_classifier')(train_x,train_y['correct'])
                models[f'{i}_{q}'] = model
                oof.loc[valid_users, q-1] = model.predict(valid_x)
            

    return models, oof


def f1_score_plot(model_name, oof, true):
    '''
    Calculate f1 score and plot f1_score vs threshold
    Args:
        model_name: used model
        oof: predict value
        true: true value
    '''
    scores=[]
    thresholds=[]

    # calculate integral f1 score
    for threshold in np.arange(0.4,0.81,0.01):
        preds = (oof.values.reshape((-1))>threshold).astype('int')
        m = f1_score(true.values.reshape((-1)),preds,average='macro')
        scores.append(m)
        thresholds.append(threshold)

    # find best threshold
    best_score = max(scores)
    best_threshold = thresholds[scores.index(best_score)]

    # plot f1_score vs threshold
    plt.figure(figsize=(20,5))
    plt.scatter([best_threshold], [best_score], color='red')
    plt.plot(thresholds, scores, color='blue')
    plt.xlabel('Threshold',size=14)
    plt.ylabel('Validation F1 Score',size=14)
    plt.title(f'{model_name} with Best F1_Score = {best_score:.4f} at Best Threshold = {best_threshold:.3}',size=18)
    plt.show()
    plt.savefig(f'{model_name}.png')


# Create ground truth table
oof = pd.DataFrame(data=np.empty((len(users),18),object),index=users)
true = oof.copy()
for q in range(1,19):
    # get true labels
    tmp = train_labels.loc[train_labels['q'] == q].loc[users]
    true[q-1] = tmp.correct.values


# Fit all features to boost_models and selected data to clf_models
boost_models = ['lgbm', 'xgb', 'random_forest', 'adaboost']
clf_models = ['knn', 'svm', 'nn']

mmodels = {}
for m in boost_models+clf_models:
    print(m)
    if m in boost_models:
        models, oof = model_training(m, df, train_labels)
    elif m in clf_models:
        models, oof = model_training(m, df_selected, train_labels)
    f1_score_plot(m, oof, true)
    mmodels[m] = models


def predict(qid, test, models):
    best_threshold = 0.63
    n_folds = 5
    val = 0
    for fold in range(n_folds):
        val += models[f'{fold}_{qid}'].predict(test.drop('level_group', axis=1))[0]
    return val > best_threshold * n_folds

sample_submission = pd.read_csv('sample_submission.csv')
test = pd.read_csv("test_feature_data.csv", index_col='session_id')

# infer test data
grp = test.level_group.values[0]
sample_submission['qid'] = sample_submission['session_id'].apply(lambda x: x.split("_")[1][1:]).astype(int)
sample_submission['correct'] = sample_submission['qid'].apply(lambda x: predict(qid, df, mmodels['lgbm'])).astype(int)
del sample_submission['qid']
sample_submission.to_csv('sample_submission.csv')
