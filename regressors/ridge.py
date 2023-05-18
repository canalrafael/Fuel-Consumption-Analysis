import os
import json
import pandas as pd

from regressor_utils import *

import xgboost as xgb
from sklearn.linear_model import Ridge

ridge = Ridge()

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
with open("data/feature_selections/ridge_stage1.json") as f:
    ridge1 = json.loads(f.read())
with open("data/feature_selections/ridge_stage2.json") as f:
    ridge2 = json.loads(f.read())
with open("data/feature_selections/ridge_stage3.json") as f:
    ridge3 = json.loads(f.read())

features = ["SFS","RFE","RFCVE","SFM"]
filterfeatures = ["Select_Percentile","Select_KBest"]

ridge_results_stage_1 = []
ridge_results_stage_2 = []
ridge_results_stage_3 = []

for feature in filterfeatures:
    print(f"Starting validations for {feature}")
    ridge_results1 = cross_tests_validate(stage1_paths,filterreg1[feature], "Consumption",ridge, f"ridge_stage1_{feature}",savefile = False)
    ridge_results1["FeatureSet"] = feature
    ridge_results_stage_1.append(ridge_results1)

    ridge_results2 = cross_tests_validate(stage2_paths,filterreg2[feature], "Consumption",ridge, f"ridge_stage2_{feature}",savefile = False)
    ridge_results2["FeatureSet"] = feature
    ridge_results_stage_2.append(ridge_results2)
    
    ridge_results3 = cross_tests_validate(stage3_paths,filterreg3[feature], "Consumption",ridge, f"ridge_stage3_{feature}",savefile = False)
    ridge_results3["FeatureSet"] = feature
    ridge_results_stage_3.append(ridge_results3)    

    

for feature in features:
    print(f"Starting validations for {feature}")
    ridge_results1 = cross_tests_validate(stage1_paths,ridge1[feature], "Consumption",ridge, f"ridge_stage1_{feature}",savefile=False)
    ridge_results1["FeatureSet"] = feature
    ridge_results_stage_1.append(ridge_results1)

    ridge_results2 = cross_tests_validate(stage2_paths,ridge2[feature], "Consumption",ridge, f"ridge_stage2_{feature}",savefile=False)
    ridge_results2["FeatureSet"] = feature
    ridge_results_stage_2.append(ridge_results2)

    ridge_results3 = cross_tests_validate(stage3_paths,ridge3[feature], "Consumption",ridge, f"ridge_stage3_{feature}",savefile=False)
    ridge_results3["FeatureSet"] = feature
    ridge_results_stage_3.append(ridge_results3)

    

pd.concat(ridge_results_stage_1).to_excel("ridge_results_stage_1.xls")
pd.concat(ridge_results_stage_2).to_excel("ridge_results_stage_2.xls")
pd.concat(ridge_results_stage_3).to_excel("ridge_results_stage_3.xls")
