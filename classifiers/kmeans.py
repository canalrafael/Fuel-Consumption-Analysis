import os
import json
import pandas as pd

from classifier_utils import *

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

stage1_paths = [
    # STAGE 1
    "data/testes/teste_04_10.parquet",
    "data/testes/teste_18_10.parquet"
]

stage2_paths = [
    # STAGE 2
    "data/testes/teste_02_01_13h56min_14h14min.parquet",
    "data/testes/teste_02_01_17h33min_18h13min.parquet"
]

stage3_paths = [
    # STAGE 3
    "data/testes/teste_09_02.parquet"
]

with open("data/feature_selections/filter_reg_stage1.json") as f:
    filterreg1 = json.loads(f.read())
with open("data/feature_selections/filter_reg_stage2.json") as f:
    filterreg2 = json.loads(f.read())
with open("data/feature_selections/filter_reg_stage3.json") as f:
    filterreg3 = json.loads(f.read())
with open("data/feature_selections/xgb_reg_stage1.json") as f:
    modelfeatures1 = json.loads(f.read())
with open("data/feature_selections/xgb_reg_stage2.json") as f:
    modelfeatures2 = json.loads(f.read())
with open("data/feature_selections/xgb_reg_stage3.json") as f:
    modelfeatures3 = json.loads(f.read())

features = ["SFS","RFE","RFCVE","SFM"]
filterfeatures = ["Select_Percentile","Select_KBest"]

kmeans_results_stage_1 = []
kmeans_results_stage_2 = []
kmeans_results_stage_3 = []

for feature in filterfeatures:
    print(f"Starting validations for {feature}")
    kmeans_results1 = cross_tests_validate(stage1_paths,filterreg1[feature], "Consumption_Threshold",kmeans, f"xgb_reg_reg_stage1_{feature}",savefile = False)
    kmeans_results1["FeatureSet"] = feature
    kmeans_results_stage_1.append(kmeans_results1)

    kmeans_results2 = cross_tests_validate(stage2_paths,filterreg2[feature], "Consumption_Threshold",kmeans, f"xgb_reg_reg_stage2_{feature}",savefile = False)
    kmeans_results2["FeatureSet"] = feature
    kmeans_results_stage_2.append(kmeans_results2)
    
    kmeans_results3 = cross_tests_validate(stage3_paths,filterreg3[feature], "Consumption_Threshold",kmeans, f"xgb_reg_reg_stage3_{feature}",savefile = False)
    kmeans_results3["FeatureSet"] = feature
    kmeans_results_stage_3.append(kmeans_results3)    

    

for feature in features:
    print(f"Starting validations for {feature}")
    kmeans_results1 = cross_tests_validate(stage1_paths,modelfeatures1[feature], "Consumption_Threshold",kmeans, f"xgb_reg_reg_stage1_{feature}",savefile=False)
    kmeans_results1["FeatureSet"] = feature
    kmeans_results_stage_1.append(kmeans_results1)

    kmeans_results2 = cross_tests_validate(stage2_paths,modelfeatures2[feature], "Consumption_Threshold",kmeans, f"xgb_reg_reg_stage2_{feature}",savefile=False)
    kmeans_results2["FeatureSet"] = feature
    kmeans_results_stage_2.append(kmeans_results2)

    kmeans_results3 = cross_tests_validate(stage3_paths,modelfeatures3[feature], "Consumption_Threshold",kmeans, f"xgb_reg_reg_stage3_{feature}",savefile=False)
    kmeans_results3["FeatureSet"] = feature
    kmeans_results_stage_3.append(kmeans_results3)

    

pd.concat(kmeans_results_stage_1).to_excel("kmeans_results_stage_1.xls")
pd.concat(kmeans_results_stage_2).to_excel("kmeans_results_stage_2.xls")
pd.concat(kmeans_results_stage_3).to_excel("kmeans_results_stage_3.xls")
