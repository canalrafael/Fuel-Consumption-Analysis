import os
import json
from iase_csm import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix

data_paths = [
    "data/testes/teste_30_09.parquet",
    "data/testes/teste_04_10.parquet",
    "data/testes/teste_18_10.parquet",
    "data/testes/teste_02_01_pt1.parquet",
    "data/testes/teste_02_01_pt2.parquet",
    "data/testes/teste_09_02.parquet",
]

data_list = list(map(treat_parquet, data_paths))

features_path = "data/feature_selections/xgboost_class_newtrain.json"

with open(features_path, "r") as f:
    json_vars = f.read()
    selected_features = json.loads(json_vars)

df_train = pd.concat(data_list)

y_test = (df_train["fuel_consumption_variable"] >= 5.5) * 1

model = KMeans(n_clusters=2)

results = []
results_dict = {}

for fs_model in selected_features:
    features = selected_features[fs_model]

    print("\n Results for " + fs_model + " features")

    x_train = df_train[features]

    model.fit(x_train)
    y_pred = model.predict(x_train)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    results.append((fs_model, accuracy, precision, recall))

results_df = pd.DataFrame(results, columns=["FS_Model", "Test Accuracy", "Test Precision", "Test Recall"])
print(results_df.head())
results_df.to_excel('data/validation_logs/xgboost_commonvars_kmeans_cluster.xls')
