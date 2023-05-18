## import
import os
import pandas as pd
import numpy as np
import json
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

def plot_regression_scores(target_test,target_predict, savefig = False, figname="",plot=True):
    error = target_test-target_predict

    fig, axes = plt.subplots(3,1)

    fig.tight_layout()

    axes[0].plot(target_test, label="Real Consume")
    axes[0].plot(target_predict, label="Predicted Consume")
    axes[0].legend()
    axes[0].set_title("Test vs. prediction comparison by timestamp")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Fuel Consumption (l/h)")
    

    axes[1].plot(error)
    axes[1].set_title("Error by timestamp")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Fuel Consumption (l/h)")

    axes[2].scatter(target_test,target_predict)
    axes[2].plot(target_test,target_test,color="red")
    axes[2].set_title("Test vs. Prediction by test values")
    axes[2].set_xlabel("Test Fuel Consumption ($(l/h)2$)")
    axes[2].set_ylabel("Predicted Fuel Consumption ($(l/h)2$)")

    fig.suptitle(f"Regression scores {figname}")

    fig.set_size_inches(18.5, 10.5)

    if plot:
        plt.show()

    if savefig:
        plt.savefig(f"data/validation_logs/regression_scores_{figname}.png")


def regression_scores(target_test, target_predict):
    #returns common scores for regression algorithms
    return {
        "mae" : metrics.mean_absolute_error(target_test, target_predict),
        "rmse" : metrics.mean_squared_error(target_test,target_predict),
        "maxerror" : metrics.max_error(target_test,target_predict),
        "r2" : metrics.r2_score(target_test,target_predict)
    }

def cross_tests_validate(path_list,features,target,model,problem_type,results_name=f"log_{time.ctime(time.time())}",savefile=True):

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

            mae = metrics.mean_absolute_error(y_test, y_pred),
            mse = metrics.mean_squared_error(y_test,y_pred),
            maxerror = metrics.max_error(y_test,y_pred),
            r2 = metrics.r2_score(y_test,y_pred)
            mae = metrics.mean_absolute_error(y_test,y_pred)

            print("MAE:", mae[0])
            print("MSE", mse[0])
            print("Max error:", maxerror[0])
            print("R2", r2)
            print("MAE", mae)

            plot_regression_scores(y_test.values,y_pred, True,f"{results_name}_Test_{test_path}_Train_{train_paths}",plot=False)

            results.append((test_path,mae[0], mse[0], maxerror[0], r2,mape))

    else:
        df_train = data_list[0]

        x_train = df_train[features]
        y_train = df_train[target]
        print(features)

        if problem_type=="classification":
            scores = cross_validate(model,x_train,y_train,cv=3,scoring=["accuracy","precision","recall"])   
            print("Accuracy:", scores["test_accuracy"])
            print("Precision:", scores["test_precision"])
            print("Recall:", scores["test_recall"])

            result_rows = [("Fold "+str(i),scores["test_accuracy"][i],scores["test_precision"][i],scores["test_recall"][i]) for i in range(3)] 
            results += result_rows
        elif problem_type=="regression":
            scores = cross_validate(model,x_train,y_train,cv=3,scoring=["neg_mean_absolute_error",'neg_root_mean_squared_error',"max_error","r2","neg_mean_absolute_percentage_error"])   

            mae = -scores["test_neg_mean_absolute_error"]
            mse = -scores["test_neg_root_mean_squared_error"]
            maxerror = scores["test_max_error"]
            r2 = scores["test_r2"]
            mape = -scores["test_neg_mean_absolute_percentage_error"]

            print("MAE:", mae)
            print("MSE", mse)
            print("Max error:", maxerror)
            print("R2", r2)
            print("MAPE", mape)

            result_rows = [("Fold "+str(i),mae[i],mse[i],maxerror[i],r2[i],mape[i]) for i in range(3)] 
            results += result_rows
    
    results_df = pd.DataFrame(results, columns=["Test","Test MAE","Test MSE","Test Max Error", "R2","MAPE"])
    
    print(results_df.head())
    if savefile:
        results_df.to_excel(f'data/validation_logs/{results_name}.xls')
    return results_df


