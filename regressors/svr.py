import os
import json
import pandas as pd

from regressor_utils import *

import xgboost as xgb
from sklearn.svm import LinearSVR

svr = LinearSVR(epsilon=1.0, max_iter=10000)


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
with open("data/feature_selections/svr_stage1.json") as f:
    svr1 = json.loads(f.read())
with open("data/feature_selections/svr_stage2.json") as f:
    svr2 = json.loads(f.read())
with open("data/feature_selections/svr_stage3.json") as f:
    svr3 = json.loads(f.read())

features = ["SFS","RFE","RFCVE","SFM"]
filterfeatures = ["Select_Percentile","Select_KBest"]

svr_results_stage_1 = []
svr_results_stage_2 = []
svr_results_stage_3 = []

for feature in filterfeatures:
    print(f"Starting validations for {feature}")
    svr_results1 = cross_tests_validate(stage1_paths,filterreg1[feature], "Consumption",svr, f"svr_stage1_{feature}",savefile = False)
    svr_results1["FeatureSet"] = feature
    svr_results_stage_1.append(svr_results1)

    svr_results2 = cross_tests_validate(stage2_paths,filterreg2[feature], "Consumption",svr, f"svr_stage2_{feature}",savefile = False)
    svr_results2["FeatureSet"] = feature
    svr_results_stage_2.append(svr_results2)
    
    svr_results3 = cross_tests_validate(stage3_paths,filterreg3[feature], "Consumption",svr, f"svr_stage3_{feature}",savefile = False)
    svr_results3["FeatureSet"] = feature
    svr_results_stage_3.append(svr_results3)    

    

for feature in features:
    print(f"Starting validations for {feature}")
    svr_results1 = cross_tests_validate(stage1_paths,svr1[feature], "Consumption",svr, f"svr_stage1_{feature}",savefile=False)
    svr_results1["FeatureSet"] = feature
    svr_results_stage_1.append(svr_results1)

    svr_results2 = cross_tests_validate(stage2_paths,svr2[feature], "Consumption",svr, f"svr_stage2_{feature}",savefile=False)
    svr_results2["FeatureSet"] = feature
    svr_results_stage_2.append(svr_results2)

    svr_results3 = cross_tests_validate(stage3_paths,svr3[feature], "Consumption",svr, f"svr_stage3_{feature}",savefile=False)
    svr_results3["FeatureSet"] = feature
    svr_results_stage_3.append(svr_results3)

    

pd.concat(svr_results_stage_1).to_excel("svr_results_stage_1.xls")
pd.concat(svr_results_stage_2).to_excel("svr_results_stage_2.xls")
pd.concat(svr_results_stage_3).to_excel("svr_results_stage_3.xls")
