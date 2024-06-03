import json
import pandas as pd

from classifier_utils import *

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)

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
with open("data/feature_selections/logreg_stage1.json") as f:
    logreg1 = json.loads(f.read())
with open("data/feature_selections/logreg_stage2.json") as f:
    logreg2 = json.loads(f.read())
with open("data/feature_selections/logreg_stage3.json") as f:
    logreg3 = json.loads(f.read())

features = ["SFS","RFE","RFCVE","SFM"]
filterfeatures = ["Select_Percentile","Select_KBest"]

logreg_results_stage_1 = []
logreg_results_stage_2 = []
logreg_results_stage_3 = []

for feature in filterfeatures:
    print(f"Starting validations for {feature}")
    logreg_results1 = cross_tests_validate(stage1_paths,filterreg1[feature], "consumption",logreg, f"xgb_reg_reg_stage1_{feature}",savefile = False)
    logreg_results1["FeatureSet"] = feature
    logreg_results_stage_1.append(logreg_results1)

    logreg_results2 = cross_tests_validate(stage2_paths,filterreg2[feature], "consumption",logreg, f"xgb_reg_reg_stage2_{feature}",savefile = False)
    logreg_results2["FeatureSet"] = feature
    logreg_results_stage_2.append(logreg_results2)
    
    logreg_results3 = cross_tests_validate(stage3_paths,filterreg3[feature], "consumption",logreg, f"xgb_reg_reg_stage3_{feature}",savefile = False)
    logreg_results3["FeatureSet"] = feature
    logreg_results_stage_3.append(logreg_results3)    

    

for feature in features:
    print(f"Starting validations for {feature}")
    logreg_results1 = cross_tests_validate(stage1_paths,logreg1[feature], "consumption",logreg, f"xgb_reg_reg_stage1_{feature}",savefile=False)
    logreg_results1["FeatureSet"] = feature
    logreg_results_stage_1.append(logreg_results1)

    logreg_results2 = cross_tests_validate(stage2_paths,logreg2[feature], "consumption",logreg, f"xgb_reg_reg_stage2_{feature}",savefile=False)
    logreg_results2["FeatureSet"] = feature
    logreg_results_stage_2.append(logreg_results2)

    logreg_results3 = cross_tests_validate(stage3_paths,logreg3[feature], "consumption",logreg, f"xgb_reg_reg_stage3_{feature}",savefile=False)
    logreg_results3["FeatureSet"] = feature
    logreg_results_stage_3.append(logreg_results3)

    

pd.concat(logreg_results_stage_1).to_excel("logreg_results_stage_1.xls")
pd.concat(logreg_results_stage_2).to_excel("logreg_results_stage_2.xls")
pd.concat(logreg_results_stage_3).to_excel("logreg_results_stage_3.xls")
