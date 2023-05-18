import os
import json
import pandas as pd

from regressor_utils import *

import xgboost as xgb

xgbreg = xgb.XGBRegressor()

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
    xgbreg1 = json.loads(f.read())
with open("data/feature_selections/xgb_reg_stage2.json") as f:
    xgbreg2 = json.loads(f.read())
with open("data/feature_selections/xgb_reg_stage3.json") as f:
    xgbreg3 = json.loads(f.read())

features = ["SFS","RFE","RFCVE","SFM"]
filterfeatures = ["Select_Percentile","Select_KBest"]

xgbreg_results_stage_1 = []
xgbreg_results_stage_2 = []
xgbreg_results_stage_3 = []

for feature in filterfeatures:
    print(f"Starting validations for {feature}")
    xgbreg_results1 = cross_tests_validate(stage1_paths,filterreg1[feature], "Consumption",xgbreg, f"xgb_reg_stage1_{feature}",savefile = False)
    xgbreg_results1["FeatureSet"] = feature
    xgbreg_results_stage_1.append(xgbreg_results1)

    xgbreg_results2 = cross_tests_validate(stage2_paths,filterreg2[feature], "Consumption",xgbreg, f"xgb_reg_stage2_{feature}",savefile = False)
    xgbreg_results2["FeatureSet"] = feature
    xgbreg_results_stage_2.append(xgbreg_results2)
    
    xgbreg_results3 = cross_tests_validate(stage3_paths,filterreg3[feature], "Consumption",xgbreg, f"xgb_reg_stage3_{feature}",savefile = False)
    xgbreg_results3["FeatureSet"] = feature
    xgbreg_results_stage_3.append(xgbreg_results3)    

    

for feature in features:
    print(f"Starting validations for {feature}")
    xgbreg_results1 = cross_tests_validate(stage1_paths,xgbreg1[feature], "Consumption",xgbreg, f"xgb_reg_stage1_{feature}",savefile=False)
    xgbreg_results1["FeatureSet"] = feature
    xgbreg_results_stage_1.append(xgbreg_results1)

    xgbreg_results2 = cross_tests_validate(stage2_paths,xgbreg2[feature], "Consumption",xgbreg, f"xgb_reg_stage2_{feature}",savefile=False)
    xgbreg_results2["FeatureSet"] = feature
    xgbreg_results_stage_2.append(xgbreg_results2)

    xgbreg_results3 = cross_tests_validate(stage3_paths,xgbreg3[feature], "Consumption",xgbreg, f"xgb_reg_stage3_{feature}",savefile=False)
    xgbreg_results3["FeatureSet"] = feature
    xgbreg_results_stage_3.append(xgbreg_results3)

    

pd.concat(xgbreg_results_stage_1).to_excel("xgbreg_results_stage_1.xls")
pd.concat(xgbreg_results_stage_2).to_excel("xgbreg_results_stage_2.xls")
pd.concat(xgbreg_results_stage_3).to_excel("xgbreg_results_stage_3.xls")
