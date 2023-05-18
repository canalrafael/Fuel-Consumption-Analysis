import os
import json
import pandas as pd

from classifier_utils import *

import xgboost as xgb

xgbclass = xgb.XGBClassifier()

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
    filterclass1 = json.loads(f.read())
with open("data/feature_selections/filter_class_stage2.json") as f:
    filterclass2 = json.loads(f.read())
with open("data/feature_selections/filter_class_stage3.json") as f:
    filterclass3 = json.loads(f.read())
with open("data/feature_selections/xgb_class_stage1.json") as f:
    xgbclass1 = json.loads(f.read())
with open("data/feature_selections/xgb_class_stage2.json") as f:
    xgbclass2 = json.loads(f.read())
with open("data/feature_selections/xgb_class_stage3.json") as f:
    xgbclass3 = json.loads(f.read())

features = ["SFS","RFE","RFCVE","SFM"]
filterfeatures = ["Select_Percentile","Select_KBest"]

xgbclass_results_stage_1 = []
xgbclass_results_stage_2 = []
xgbclass_results_stage_3 = []

for feature in filterfeatures:
    print(f"Starting validations for {feature}")
    xgbclass_results1 = cross_tests_validate(stage1_paths,filterclass1[feature], "Consumption_Threshold",xgbclass, f"xgb_class_stage1_{feature}",savefile = False)
    xgbclass_results1["FeatureSet"] = feature
    xgbclass_results_stage_1.append(xgbclass_results1)

    xgbclass_results2 = cross_tests_validate(stage2_paths,filterclass2[feature], "Consumption_Threshold",xgbclass, f"xgb_class_stage2_{feature}",savefile = False)
    xgbclass_results2["FeatureSet"] = feature
    xgbclass_results_stage_2.append(xgbclass_results2)
    
    xgbclass_results3 = cross_tests_validate(stage3_paths,filterclass3[feature], "Consumption_Threshold",xgbclass, f"xgb_class_stage3_{feature}",savefile = False)
    xgbclass_results3["FeatureSet"] = feature
    xgbclass_results_stage_3.append(xgbclass_results3)    

    

for feature in features:
    print(f"Starting validations for {feature}")
    xgbclass_results1 = cross_tests_validate(stage1_paths,xgbclass1[feature], "Consumption_Threshold",xgbclass, f"xgb_class_stage1_{feature}",savefile=False)
    xgbclass_results1["FeatureSet"] = feature
    xgbclass_results_stage_1.append(xgbclass_results1)

    xgbclass_results2 = cross_tests_validate(stage2_paths,xgbclass2[feature], "Consumption_Threshold",xgbclass, f"xgb_class_stage2_{feature}",savefile=False)
    xgbclass_results2["FeatureSet"] = feature
    xgbclass_results_stage_2.append(xgbclass_results2)

    xgbclass_results3 = cross_tests_validate(stage3_paths,xgbclass3[feature], "Consumption_Threshold",xgbclass, f"xgb_class_stage3_{feature}",savefile=False)
    xgbclass_results3["FeatureSet"] = feature
    xgbclass_results_stage_3.append(xgbclass_results3)

    

pd.concat(xgbclass_results_stage_1).to_excel("xgbclass_results_stage_1.xls")
pd.concat(xgbclass_results_stage_2).to_excel("xgbclass_results_stage_2.xls")
pd.concat(xgbclass_results_stage_3).to_excel("xgbclass_results_stage_3.xls")
