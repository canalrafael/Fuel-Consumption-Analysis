import os
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

from sklearn.feature_selection import mutual_info_regression


from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest, r_regression
from  sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

def split_feature_target(df : pd.DataFrame, target_var : str):
    #split df in one Series with the target var and an df with remaining vars

    target_col = df[target_var]
    feature_cols = df.drop(columns=[target_var])
    return target_col, feature_cols


def select_percentil(feature_train, target_train,perc):
    selected_top_columns = SelectPercentile(mutual_info_regression, percentile=perc)
    selected_top_columns.fit(feature_train, target_train)

    return list(feature_train.columns[selected_top_columns.get_support()].values)

def select_kbest(feature_train, target_train,k):
    sel_ten_cols = SelectKBest(r_regression, k=k)

    #the r_regression yields some inf or NaN values
    #wich happens when the std is 0, because the expression
    #involves division by the std, so included the step below to avoid it
    feature_train = feature_train[feature_train.columns[feature_train.std()!=0]]

    sel_ten_cols.fit(feature_train, target_train)
    return list(feature_train.columns[sel_ten_cols.get_support()].values)

def select_sfs(model, feature_train, target_train):
    sfs1 = sfs(model, n_features_to_select=5, scoring='neg_mean_squared_error', direction='forward', cv = 10)
    sfs1 = sfs1.fit(feature_train, target_train)
    return list(sfs1.get_feature_names_out())

def select_rfe(model, feature_train, target_train):
    rfe_selector = RFE(estimator=model, n_features_to_select=7, step=3, verbose=5)
    rfe_selector.fit(feature_train, target_train)
    rfe_support = rfe_selector.get_support()
    return feature_train.loc[:,rfe_support].columns.tolist()


def select_rfcve(model, feature_train, target_train):
    rf_selector = RFECV(model, min_features_to_select=5, cv =10)
    rf_selector.fit(feature_train, target_train)
    rf_support = rf_selector.get_support()
    return feature_train.loc[:,rf_support].columns.tolist()

def select_sfm(model, feature_train, target_train):
    features_classifier = SelectFromModel(model,max_features=5)
    features_classifier.fit(feature_train, target_train)
    return list(feature_train.columns[features_classifier.get_support()])

def modelbased_feature_selection(model, x_train, y_train,savejson=True, filename='fs'):
    selectedFeaturesDict = {}

    print("Starting selection by SFS")
    selectedFeaturesDict["SFS"] = select_sfs(model,x_train,y_train)
    print("Starting selection by Select RFE")
    selectedFeaturesDict["RFE"] = select_rfe(model,x_train,y_train)
    print("Starting selection by RFCVE")
    selectedFeaturesDict["RFCVE"] = select_rfcve(model,x_train,y_train)
    print("Starting selection by SFM")
    selectedFeaturesDict["SFM"] = select_sfm(model,x_train,y_train)

    if savejson:
        with open(f"data/feature_selections/{filename}.json","w") as f:
            print(selectedFeaturesDict)
            f.write(json.dumps(selectedFeaturesDict))
        print(f"Json saved in data/feature_selections/{filename}.json")

    return selectedFeaturesDict

def filter_feature_selection(x_train, y_train,savejson=True, filename='fs'):
    selectedFeaturesDict = {}

    print("Starting selection by Select Percentile")
    selectedFeaturesDict["Select_Percentile"] = select_percentil(x_train, y_train,15)
    print("Starting selection by Select KBest")
    selectedFeaturesDict["Select_KBest"] = select_kbest(x_train, y_train,7)

    if savejson:
        with open(f"data/feature_selections/{filename}.json","w") as f:
            print(selectedFeaturesDict)
            f.write(json.dumps(selectedFeaturesDict))
        print(f"Json saved in data/feature_selections/{filename}.json")

    return selectedFeaturesDict