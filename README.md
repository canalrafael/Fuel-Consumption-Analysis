# Fuel-Consumption-Analysis

This git contains the files used for the paper "Machine learning for real-time fuel consumption prediction and driving profile
classification based on ECU data", by Rafael Canal, Felipe Kaminsky Riffel and Giovani Gracioli.

## Data Stucture

In our work, the datasets were structured having one separated file for each performed experiment. The server exported the data in the `parquet` format, which may be easily imported as a DataFrame by `pandas` library. These experiments are set to be saved in `data/testes`.

After importing, the data consists essentially of a DataFrame where the rows are the data records and each column is the respective feature.

Results of algorithm’s returns are set to be saved on `data` folder.

## Instructions for Feature Selection

On `feature_selectors/fs_utils.py` we joined all the routines used to apply feature selection on our work. We split one function for every method applied, in order to get an uniformed formatted return for all of them: a `list` with the keys for each feature. Beyond these, we congregate the used methods in two functions which perform the selection with all methods and save the formatted results, given some dataset and/or a model: `model_feature_selection` and `filter_feature_selection`.


We created two different routines because of the differences in the types of feature selectors work principles. This distinction is mainly on the necessity or not in using the predictor on the feature selection process.

Briefly, the main types are:

- Filter Type feature selectors, which perform the selection based on statistical measures between train and target features, without the need of the model. In our case, we use the Select Percentile and Select K-Best filter method.
- Wrapped and Embedded feature selectors, which use iterative processes of training and testing the model with different sets of available features, evaluating and ranking the features based on these train/test results. In this case, the final prediction model is needed to perform the selection. On our context, the used Wrapped and Embedded methods were Select from Model (SFM), Sequential Feature Selection(SFS), Recursive Feature Elimination (RFE) and Recursive Feature Elimination with Cross-Validation (RFECV).

All implementations used were the ones provided by `sklearn.feature_selection` library. Below, a description of the received parameters in both methods. Both return a dictionary, where each key is the used selection method and the respective list of selected features.

```
filter_feature_selection
    x_train: DataFrame or np.array.    Array of features;
    y_train: DataFrame ou np.array.    Target value array;
    save_jason=true: Boolean    If you want to save an JSON file with the dictionary information.
    save_excel=true: Boolean    If you want to save a sheet file wth the dictionary information.
    filename = "fs": String.     File name to be saved, same for JSON and Excel.


modelbased_feature_selection:
    model: Object   	 Predictor model object, not fitted. It needs to be on the Estimator API format of Sklearn.
    x_train: DataFrame or np.array.    Array of features;
    y_train: DataFrame ou np.array.    Target value array;
    save_jason=true: Boolean    If you want to save an JSON file with the dictionary information.
    save_excel=true: Boolean    If you want to save a sheet file wth the dictionary information.
    filename = "fs": String.     File name to be saved, same for JSON and Excel.
```

An example of how these methods were applied is on `feature_selectors/fs.py` file.

## Instructions for model evaluation

As described in our work, the experiments were divided on Stages, because of the differences on the available features between each Stage of experiments. With this, we performed a cross-validation for both classifier and regression methods consisting in: taking one experiment as test set, and the remaining experiment's data of the respective Stage as training set, performing a train/test in each round for every experiment set in the Stage. Furthermore, in early stages of development, we considered the case where there was only one experiment in a Stage. In that case, we performed a usual Cross-Validation, splitting the dataset in random fold sets using the `sklearn.model_selection.cross_validate` method.

We implemented routines for performing the described process independent of dataset and informed model: `cross_tests_validate`. Besides of having the same cross-validation process, we split classifier and regression in two different methods because of the different score outputs between each type of prediction. Both have the same name `cross_tests_validate`, but are available in two different files: `classifiers/classifier_utils.py` and `regression/regression_utils.py`. Beyond this method, each file contains another useful methods, as automated fitting/prediction, plotting and scoring.


Below, the description of the parameters informed to `cross_tests_validate` methods. Examples of applications are available on `classifiers` and `regression` folders, splitting one file for each model used.

```
cross_tests_validate:
    path_list: list   		 List with the paths for each experiment file.
    features: list or array,   	 List with the keys for features, relative to the informed dataset
    target: str or int,   		 Key for the target value on each dataset
    model: Object:   				 Predictor model object, not fitted. It needs to be on the Estimator API format of Sklearn
    results_name: str=f"log_{time.ctime(time.time())}",    Name of the output file.
    savefile: Boolean=True   	 If you want to save a excel file with the results. Default is True, being saved to file data/validation_logs/results_name.xls
```
