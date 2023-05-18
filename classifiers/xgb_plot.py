import os
import json
import numpy as np
import xgboost as xgb
from iase_csm import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

data_paths = [
    "data/testes/teste_30_09.parquet",
    "data/testes/teste_04_10.parquet",
    "data/testes/teste_18_10.parquet",
    "data/testes/teste_02_01_pt1.parquet",
    "data/testes/teste_02_01_pt2.parquet",
    "data/testes/teste_09_02.parquet",
]

data_list = list(map(treat_parquet,data_paths))

features_path = "data/feature_selections/xgboost_class_latestexperiments.json"

with open(features_path,"r") as f:
    json_vars = f.read()
    selected_features = json.loads(json_vars)

model = xgb.XGBClassifier()

results = []
results_dict = {}


fs_model = "XGB_RFE"

features = selected_features[fs_model]

print("\n Results for " + fs_model + " features")

if len(data_paths)>1:
    for i in range(len(data_list)):
        test_path = data_paths[i]
        print("\n Test:",os.path.split(test_path)[1], "\n")
        print("\n Train:",",".join(map(lambda x: os.path.split(x)[1],data_paths[:i]+data_paths[i+1:])))

        df_test = data_list[i]
        df_train = pd.concat(data_list[:i]+data_list[i+1:])

        x_train = df_train[features]
        y_train = (df_train["fuel_consumption_variable"] >= 5.5)*1

        x_test = df_test[features]
        y_test = (df_test["fuel_consumption_variable"] >= 5.5)*1

        y_pred = fit_predict(model,x_train, y_train,x_test)

        #Graph plot
        plt.figure(figsize=(8,3))
        plt.plot(df_test["fuel_consumption_variable"].values,label="Consumption")
        plt.plot(5.5*np.ones(len(y_pred)),'--',label = "Threshold",color="blue")
        plt.fill_between(np.arange(len(y_pred)),0,df_test["fuel_consumption_variable"].max(),where = y_pred,facecolor='red', alpha=0.3)
        plt.legend()
        plt.ylabel("Fuel consumption (l/h)")
        plt.xlabel("Time (s)")
        plt.xticks([])
        plt.show()

        accuracy = metrics.accuracy_score(y_test,y_pred)
        precision = metrics.precision_score(y_test,y_pred)
        recall = metrics.recall_score(y_test,y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)

        results.append((fs_model, os.path.split(test_path)[1],accuracy,precision,recall))
else:
    df_train = treat_parquet(data_paths[0])

    x_train = df_train[features]
    y_train = (df_train["fuel_consumption_variable"] >= 5.5)*1

    scores = cross_validate(model,x_train,y_train,cv=4,scoring=["accuracy","precision","recall"])

    print("Accuracy:", scores["test_accuracy"])
    print("Precision:", scores["test_precision"])
    print("Recall:", scores["test_recall"])

    result_rows = [(fs_model,"Fold "+str(i),scores["test_accuracy"][i],scores["test_precision"][i],scores["test_recall"][i]) for i in range(4)]
    results += result_rows

results_df = pd.DataFrame(results, columns=["FS_Model","Test","Test Accuracy","Test Precision","Test Recall"])
print(results_df.head())
