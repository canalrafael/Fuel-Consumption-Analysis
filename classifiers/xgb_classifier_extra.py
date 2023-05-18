import os
import json
import numpy as np
import xgboost as xgb
from iase_csm import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
import sklearn

data_paths = [
    "data/testes/teste_30_09.parquet",
    "data/testes/teste_04_10.parquet",
    "data/testes/teste_18_10.parquet",
    "data/testes/teste_02_01_pt1.parquet",
    "data/testes/teste_02_01_pt2.parquet",
]

test_list = list(map(treat_parquet, test_paths))

features_path = "data/feature_selections/xgboost_class_newtrain.json"

with open(features_path, "r") as f:
    json_vars = f.read()
    selected_features = json.loads(json_vars)

df_train = treat_parquet("data/testes/teste_09_02.parquet")
y_train = (df_train["fuel_consumption_variable"] >= 5.5) * 1

xgb_class = xgb.XGBClassifier()

results = []
results_dict = {}

for fs_model in selected_features:
    features = selected_features[fs_model]
    x_train = df_train[features]

    print(x_train.columns.values)
    xgb_class.fit(x_train, y_train)

    print("\n Results for " + fs_model + " features")

    if len(test_paths) > 1:
        for i in range(len(test_paths)):
            test_path = test_paths[i]
            print("\n Test:", os.path.split(test_path)[1], "\n")
            print("\n Train:", ",".join(map(lambda x: os.path.split(x)[1], test_paths[:i] + test_paths[i + 1:])))

            df_test = test_list[i]

            x_test = df_test[features]
            y_test = (df_test["fuel_consumption_variable"] >= 5.5) * 1

            y_pred = xgb_class.predict(x_test)

            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)

            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)

            results.append((fs_model, os.path.split(test_path)[1], accuracy, precision, recall))
    else:
        df_train = treat_parquet(test_paths[0])

        x_train = df_train[features]
        y_train = (df_train["fuel_consumption_variable"] >= 5.5) * 1

        scores = cross_validate(xgb_class, x_train, y_train, cv=4, scoring=["accuracy", "precision", "recall"])

        print("Accuracy:", scores["test_accuracy"])
        print("Precision:", scores["test_precision"])
        print("Recall:", scores["test_recall"])

        result_rows = [(fs_model, "Fold " + str(i), scores["test_accuracy"][i], scores["test_precision"][i],
                        scores["test_recall"][i]) for i in range(4)]
        results += result_rows

results_df = pd.DataFrame(results, columns=["FS_Model", "Test", "Test Accuracy", "Test Precision", "Test Recall"])
print(results_df.head())
results_df.to_excel('data/validation_logs/xgboost_class_newexperiments_08_03.xls')
