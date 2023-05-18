#Kernel usado: IPython 3.9.13 64-bit
import pandas as pd

from fs_utils import *

import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR

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

df1 = pd.concat(map(pd.read_parquet,stage1_paths))
df2 = pd.concat(map(pd.read_parquet,stage2_paths))
df3 = pd.concat(map(pd.read_parquet,stage3_paths))

y_train_1, x_train_1 = split_feature_target(df1,"Consumption")
y_train_2, x_train_2 = split_feature_target(df2,"Consumption")
y_train_3, x_train_3 = split_feature_target(df3,"Consumption")

xgb_reg = xgb.XGBRegressor()
ridge_reg = Ridge(alpha=1)
svr = LinearSVR(epsilon=1.0,max_iter=10000)


# modelbased_feature_selection(ridge_reg, x_train_1, y_train_1,filename = "ridge_reg_stage1")
filter_feature_selection(x_train_1, y_train_1, filename = "filter_reg_stage1")
filter_feature_selection(x_train_2, y_train_2, filename = "filter_reg_stage2")
filter_feature_selection(x_train_3, y_train_3, filename = "filter_reg_stage3")
modelbased_feature_selection(xgb_reg, x_train_1, y_train_1,filename = "xgb_reg_stage1")
modelbased_feature_selection(xgb_reg, x_train_2, y_train_2,filename = "xgb_reg_stage2")
modelbased_feature_selection(xgb_reg, x_train_3, y_train_3,filename = "xgb_reg_stage3")
modelbased_feature_selection(svr, x_train_1, y_train_1,filename = "svr_reg_stage1")
modelbased_feature_selection(svr, x_train_2, y_train_2,filename = "svr_reg_stage2")
modelbased_feature_selection(svr, x_train_3, y_train_3,filename = "svr_reg_stage3")
modelbased_feature_selection(ridge_reg, x_train_1, y_train_1,filename = "ridge_reg_stage1")
modelbased_feature_selection(ridge_reg, x_train_2, y_train_2,filename = "ridge_reg_stage2")
modelbased_feature_selection(ridge_reg, x_train_3, y_train_3,filename = "ridge_reg_stage3")