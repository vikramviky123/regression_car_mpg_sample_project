import os

import copy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import shapiro

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from sklearn.inspection import permutation_importance

from sklearn import metrics

import joblib
import pickle

from typing import List, Dict, Set

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
gld = Style.BRIGHT + Fore.YELLOW
grn = Style.BRIGHT + Fore.GREEN
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
white = Style.BRIGHT + Fore.WHITE
cyan = Style.BRIGHT + Fore.CYAN
magenta = Style.BRIGHT + Fore.MAGENTA
res = Style.RESET_ALL

plt.rcParams['figure.dpi'] = 800
plt.rcParams['savefig.dpi'] = 800

sns.set(rc={"figure.dpi": 800, 'savefig.dpi': 800})
sns.set_context('notebook')
sns.set_style("ticks")

font_title = {
    'size': 20,
    'weight': 'bold',  # You can use 'weight' here instead of 'fontweight'
    'color': 'brown',
    'fontfamily': 'serif'
}
font_sub_title = {'family': 'serif',
                  'color':  'brown',
                  'weight': 'bold',
                  'size': 16,
                  }
font_label = {'family': 'serif',
              'color':  'brown',
              'weight': 'bold',
              'size': 16,
              }


def preProcess(df_act:pd.DataFrame,df:pd.DataFrame, scale:str="yes")-> pd.DataFrame:
    """Encodes object with Label Encoder.
    
    Args: 
        df (DataFrame): This DatFrame is used to for object label encoding. 
        scale    (str): To Scale the data. 
 
    Returns: 
        DataFrame: label Encoded dataframe
        Simply label encodes object attributes.        
    """
    
    df_encoded = df.copy()
    
    # Encoding the object attributes
    label_encoder = LabelEncoder()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            label_encoder.fit(df_act[col])
            df_act[col] = label_encoder.transform(df_act[col])
            df_encoded[col] = label_encoder.transform(df_encoded[col])
            
    # Create a pipeline with SimpleImputer and StandardScaler
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # You can choose other strategies for imputation
        ('scaler', StandardScaler())
                             ])
            
    # Imputing the Null values with SimpleImputer
    if scale =="yes":
        pipeline.fit(df_act)
        df_imputed = pd.DataFrame(pipeline.transform(df_encoded), columns=df_encoded.columns)
        print("Data is Imputed and scaled")
        print("-"*80)
    else:    
        simp = SimpleImputer(strategy='mean')
        simp.fit(df_act)
        df_imputed = pd.DataFrame(simp.transform(df_encoded), columns=df_encoded.columns)
        print("Data is Imputed")
        print("-"*80)
    return df_imputed


def model_evaluation(model_score: Dict, N_SPLITS: int, idx: int = 0) -> None:
    """Plots Model evaluation results.

    Args: 
        model_score (Dict): This Model results dictionary for a metric. 
        N_SPLITS     (int): number of folds used. 
        idx          (int): 0 for validation and 1 for test results. 

    Returns: 
        None 
        Simply Plots model eval results on validation data set.         
    """

    metric_list = ['r2-square', 'mae', 'rmse']
    nm = 'VALIDATION'
    metric_nm = metric_list[idx]

    fig, ax = plt.subplots(figsize=(15, 5))
    for keys in model_score.keys():
        ax.plot(model_score[keys]['valid'][metric_nm],
                label=str(keys), marker=".")

    ax.set_title(
        f"{metric_nm.upper()} scores for Each FOLD - {nm}".upper(), fontdict=font_sub_title)
    ax.set_xticks(np.arange(N_SPLITS))
    ax.set_xlabel("FOLD", fontdict=font_label)
    ax.set_ylabel(f"{metric_nm.upper()} Scores", fontdict=font_label)
    ax.tick_params(labelsize=16, labelcolor='brown', width=4)
    ax.legend()
    plt.tight_layout()

    col_name = 'Model_Evaluation'+"_"+str(nm) + "_"+str(metric_nm)
    path = "static/model_plots/"+col_name+".jpg"
    plt.savefig(path, bbox_inches="tight")

    plt.show()


def model_predict(model_dict: Dict, train_X, train_y, X_test, y_test, N_SPLITS: int, RANDOM_STATE: int) -> Dict:
    """Gets Model evaluation results.

    Args: 
        model_dict (Dict): This is Dictinary of models to be used for prediction. 
        X     (DataFrame): Independant train data. 
        y    (DataSeries): Dependant train target data. 
        N_SPLITS    (int): Number of splits for Kfold cross validation. 
        RANDOM_STATE(int): Random state for to draw same samples. 

    Returns: 
        Dict : model evaluation results 
        Simply get validation results for models used on train data set.         
    """

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    model_score = {}
    trained_models = {}

    for keys in model_dict.keys():

        model_score[keys] = {}

        model_score[keys]['valid'] = {'r2-square': [],
                                      'mae': [],
                                      'rmse': []
                                      }
        model_score[keys]['test'] = {'r2-square': [],
                                     'mae': [],
                                     'rmse': []
                                     }
        trained_models[keys] = []
        y_test_pred = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_X, train_y)):
            X_train, X_val = train_X.iloc[train_idx], train_X.iloc[val_idx]
            y_train, y_val = train_y.iloc[train_idx].to_numpy(
            ).ravel(), train_y.iloc[val_idx].to_numpy().ravel()

            # Each model should have its own instance
            model = copy.deepcopy(model_dict[keys])
            # model = model_dict[keys].__class__()

            model_fit = model.fit(X_train, y_train)
            trained_models[keys].append(model_fit)

            y_val_pred = model_fit.predict(X_val)
            model_score[keys]['valid']['r2-square'].append(
                metrics.r2_score(y_val, y_val_pred))
            model_score[keys]['valid']['mae'].append(
                metrics.mean_absolute_error(y_val, y_val_pred))
            model_score[keys]['valid']['rmse'].append(
                np.sqrt(metrics.mean_squared_error(y_val, y_val_pred)))

            y_test_pred += model_fit.predict(X_test)/N_SPLITS

        model_score[keys]['test']['r2-square'].append(
            metrics.r2_score(y_test, y_test_pred))
        model_score[keys]['test']['mae'].append(
            metrics.mean_absolute_error(y_test, y_test_pred))
        model_score[keys]['test']['rmse'].append(
            np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

    mean_score = [[], [], []]
    print("-"*34+"R2      MAE     RMSE"+"-"*14+"R2      MAE     RMSE")
    for keys in model_score.keys():
        val_r2 = np.mean(model_score[keys]['valid']['r2-square'])
        val_mae = np.mean(model_score[keys]['valid']['mae'])
        val_rmse = np.mean(model_score[keys]['valid']['rmse'])

        tst_r2 = np.mean(model_score[keys]['test']['r2-square'])
        tst_mae = np.mean(model_score[keys]['test']['mae'])
        tst_rmse = np.mean(model_score[keys]['test']['rmse'])

        print(f"{blu}Model : {gld}{keys:>{12}} {blu}Validation : {magenta}{val_r2:.4f}, {val_mae:.4f}, {val_rmse:.4f}  |  {blu}Test : {magenta}{tst_r2:.4f}, {tst_mae:.4f}, {tst_rmse:.4f}{res}")
        mean_score[0].append(val_r2)
        mean_score[1].append(val_mae)
        mean_score[2].append(val_rmse)

    metric_list = ['r2-square', 'mae', 'rmse']
    print("-"*80)
    for i in range(len(mean_score)):
        if i == 0:
            m_index = np.argmax(mean_score[i])
        else:
            m_index = np.argmin(mean_score[i])

        best_model = list(model_dict.keys())[m_index]
        best_score = mean_score[i][m_index]
        print(
            f"For Metric {gld}{metric_list[i]:>{10}} {magenta}Best Model : {best_model}   |  Best Score : {gld}{best_score:.4f}{res}")

    return model_score, trained_models


def pickle_model(trained_models: Dict,
                 directory: str = "models",
                 pickle_name: str = "pickled_model",
                 use: str = 'pickle'):
    """Creates a directory for pickled model files and saves the pickle files
        and also create pickle files from passed dictionary of models.

    Args: 
        trained_models    (Dict): This is Dictinary of models to be used for pickling. 
        directory    (DataFrame): Directory name for creating it. 
        pickle_name (DataSeries): Name of the pickled file of dict of models. 
        use                (int): Number of plits for Kfold cross validation. 

    Returns: 
        Dict : Pickled models
        Simply pickles model and loads the pickled model.         
    """

    directory = "models"
    parent_dir = os.getcwd()

    path = os.path.join(parent_dir, directory)
    print(path)
    if not os.path.exists(path):
        os.mkdir(path=path)

    print("-"*80)
    print(f"Directory with {red}{directory}{res} name created")
    print("-"*80)

    if use == 'pickle':
        pickle_file_path = directory+'/'+pickle_name+'.pkl'
        # Use pickle to save the model
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(trained_models, pickle_file)

        print(
            f"Model file pickled & SAVED in directory {red}{pickle_file_path}{res}")

        # Load the saved list of models using pickle
        with open(pickle_file_path, 'rb') as pickle_file:
            loaded_models_pickle = pickle.load(pickle_file)

        print(f"{red}pickle file{res} named {pickle_name} {red}LOADED{res}")
        print("-"*80)

        return loaded_models_pickle

    else:
        # Specify the file path where you want to save the models
        joblib_file_path = directory+'/'+pickle_name+'.pkl'
        # Use joblib to save the list of trained models to a pickle file
        joblib.dump(trained_models, joblib_file_path)
        print(f"Joblib SAVED model in file path {red}{joblib_file_path}{res}")

        # Load the saved model
        loaded_model_job = joblib.load(joblib_file_path)

        print(f"{red}Joblib file{res} named {pickle_name} {red}LOADED{res}")
        print("-"*80)

        return loaded_model_job
