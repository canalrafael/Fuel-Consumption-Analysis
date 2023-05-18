import os
import json
import pandas as pd
import time

from regressor_utils import *
from sklearn.neural_network import MLPRegressor
from fireTS.models import NARX

def narx_cross_validate(path_list,features,target,problem_type,results_name=f"log_{time.ctime(time.time())}",savefile=True):

    data_list = list(map(pd.read_parquet,path_list))
    
    results = []
    if len(data_list)>1:
        for i in range(len(data_list)):
            test_path = os.path.split(path_list[i])[1].replace(".parquet",'')
            train_paths = ",".join(map(lambda x: os.path.split(x)[1],path_list[:i]+path_list[i+1:])).replace('.parquet','')
            print("\n Test:",test_path, "\n")
            print("Train:",train_paths)

            df_test = data_list[i]
            df_train = pd.concat(data_list[:i]+data_list[i+1:])

            x_train = df_train[features]
            y_train = df_train[target]

            x_test = df_test[features]
            y_test = df_test[target]

            delay = 5

            mdl1 = NARX(
                MLPRegressor(hidden_layer_sizes=(75),solver="adam",max_iter=2000),
                auto_order=delay,
                exog_order=x_train.shape[1]*[delay])
            mdl1.fit(x_train, y_train)

            y_pred = mdl1.forecast(x_test[:delay],y_test[:delay],step = y_test.shape[0]-delay, X_future=x_test[delay:y_test.shape[0]-1])
            
            mae = metrics.mean_absolute_error(y_test[delay:], y_pred),
            mse = metrics.mean_squared_error(y_test[delay:],y_pred),
            maxerror = metrics.max_error(y_test[delay:],y_pred),
            r2 = metrics.r2_score(y_test[delay:],y_pred)
            mae = metrics.mean_absolute_error(y_test[delay:],y_pred)

            print("MAE:", mae)
            print("MSE", mse)
            print("Max error:", maxerror)
            print("R2", r2)
            print("MAE", mae)

            plot_regression_scores(y_test[delay:].values,y_pred, True,f"{results_name}_Test_{test_path}_Train_{train_paths}",plot=False)

            results.append((test_path,mae, mse, maxerror, r2))

    else:        
        for i in range(3):
            xshape = data_list[0].shape

            df_test = data_list[0][i*(xshape[0]//3):(i+1)*(xshape[0]//3)]
            df_train = pd.concat([
                data_list[0][:i*(xshape[0]//3)],
                data_list[0][(i+1)*(xshape[0]//3):],
            ])

            x_train = df_train[features]
            y_train = df_train[target]

            x_test = df_test[features]
            y_test = df_test[target]

            delay = 5

            mdl1 = NARX(
                MLPRegressor(hidden_layer_sizes=(75),solver="adam",max_iter=2000),
                auto_order=delay,
                exog_order=x_train.shape[1]*[delay])
            mdl1.fit(x_train, y_train)

            y_pred = mdl1.forecast(x_test[:delay],y_test[:delay],step = y_test.shape[0]-delay, X_future=x_test[delay:y_test.shape[0]-1])
            
            mae = metrics.mean_absolute_error(y_test[delay:], y_pred),
            mse = metrics.mean_squared_error(y_test[delay:],y_pred),
            maxerror = metrics.max_error(y_test[delay:],y_pred),
            r2 = metrics.r2_score(y_test[delay:],y_pred)
            mae = metrics.mean_absolute_error(y_test[delay:],y_pred)

            print("MAE:", mae)
            print("MSE", mse)
            print("Max error:", maxerror)
            print("R2", r2)
            print("MAE", mae)

            results.append((f"Fold {i}",mae, mse, maxerror, r2))

    if problem_type=="regression":
        results_df = pd.DataFrame(results, columns=["Test","Test MAE","Test MSE","Test Max Error", "R2"])
    
    print(results_df.head())
    if savefile:
        results_df.to_excel(f'data/validation_logs/{results_name}.xls')
    return results_df

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

delay = 5

narx_results_stage_1 = []
narx_results_stage_2 = []
narx_results_stage_3 = []

features = ["SFS","RFE","RFCVE","SFM"]
filterfeatures = ["Select_Percentile","Select_KBest"]



for feature in features:
    print(f"Starting validations for {feature}")
    narx_results1 = narx_cross_validate(stage1_paths,xgbreg1[feature], "Consumption","regression", f"xgb_reg_reg_stage1_{feature}",savefile=False)
    narx_results1["FeatureSet"] = feature
    narx_results_stage_1.append(narx_results1)

    narx_results2 = narx_cross_validate(stage2_paths,xgbreg2[feature], "Consumption","regression", f"xgb_reg_reg_stage2_{feature}",savefile=False)
    narx_results2["FeatureSet"] = feature
    narx_results_stage_2.append(narx_results2)

    narx_results3 = narx_cross_validate(stage3_paths,xgbreg3[feature], "Consumption","regression", f"xgb_reg_reg_stage3_{feature}",savefile=False)
    narx_results3["FeatureSet"] = feature
    narx_results_stage_3.append(narx_results3)


for feature in filterfeatures:
    print(f"Starting validations for {feature}")
    narx_results1 = narx_cross_validate(stage1_paths,filterreg1[feature], "Consumption","regression", f"xgb_reg_reg_stage1_{feature}",savefile=False)
    narx_results1["FeatureSet"] = feature
    narx_results_stage_1.append(narx_results1)

    narx_results2 = narx_cross_validate(stage2_paths,filterreg2[feature], "Consumption","regression", f"xgb_reg_reg_stage2_{feature}",savefile=False)
    narx_results2["FeatureSet"] = feature
    narx_results_stage_2.append(narx_results2)

    narx_results3 = narx_cross_validate(stage3_paths,filterreg3[feature], "Consumption","regression", f"xgb_reg_reg_stage3_{feature}",savefile=False)
    narx_results3["FeatureSet"] = feature
    narx_results_stage_3.append(narx_results3)


pd.concat(narx_results_stage_1).to_excel("narx_results_stage_1.xls")
pd.concat(narx_results_stage_2).to_excel("narx_results_stage_2.xls")
pd.concat(narx_results_stage_3).to_excel("narx_results_stage_3.xls")