import os
import pandas as pd
import time
from matplotlib import pyplot as plt
from sklearn import metrics

from sklearn.model_selection import cross_validate

def fit_predict(model,feature_train, target_train,feature_test=None): 
    #fit model and return prediction array of given test set
    #some models has fit_predict method, but not all of them, so made this as a warranty

    model.fit(feature_train,target_train)
    if feature_test is None:
        feature_test = feature_train

    return model.predict(feature_test)


def cross_tests_validate(path_list,features,target,model,results_name=f"log_{time.ctime(time.time())}",savefile=True):

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

            y_pred = fit_predict(model,x_train, y_train,x_test)

        
            accuracy = metrics.accuracy_score(y_test,y_pred)
            precision = metrics.precision_score(y_test,y_pred)
            recall = metrics.recall_score(y_test,y_pred)

            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)

            results.append((test_path,accuracy,precision,recall))
    else:
        df_train = data_list[0]

        x_train = df_train[features]
        y_train = df_train[target]
        print(features)

        scores = cross_validate(model,x_train,y_train,cv=3,scoring=["accuracy","precision","recall"])   
        print("Accuracy:", scores["test_accuracy"])
        print("Precision:", scores["test_precision"])
        print("Recall:", scores["test_recall"])

        result_rows = [("Fold "+str(i),scores["test_accuracy"][i],scores["test_precision"][i],scores["test_recall"][i]) for i in range(3)] 
        results += result_rows

    results_df = pd.DataFrame(results, columns=["Test","Test Accuracy","Test Precision","Test Recall"])
    
    print(results_df.head())
    if savefile:
        results_df.to_excel(f'data/validation_logs/{results_name}.xls')
    return results_df
