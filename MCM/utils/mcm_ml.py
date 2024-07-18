from sklearn import metrics
import numpy as np 
import pandas as pd 
from colorama import Fore, Back, Style, init 

def regression_report(model_name, y_true, y_pred, to_print=True):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = round(np.sqrt(mse), 4)
    r2 = metrics.r2_score(y_true, y_pred)
    
    report = dict()
    report['model']=model_name
    report['explained_variance'] = explained_variance
    report['MAE'] = mae 
    report['MSE'] = mse 
    report['r2'] = r2 
    report['RMSE'] = rmse
    
    if to_print==True:
        print(Fore.WHITE + f"============= {model_name} =============")
        print(Fore.WHITE + "explained_variance   :", round(explained_variance, 4))
        print(Fore.GREEN + '\t\tr2   :', str(round(r2, 4)))
        print(Fore.WHITE + '\t\tMAE  :', str(round(mae, 4)))
        print(Fore.WHITE + '\t\tMSE  :', str(round(mse, 4)))
        print(Fore.RED   + '\t\tRMSE :', str(round(rmse, 4)))
        
    return report

#####################################################################################################
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR 

import warnings 
warnings.filterwarnings('ignore')

def ML_BaseLineTest(train_Xy, test_Xy):
    train_X, train_y = train_Xy.iloc[:, :-1], train_Xy.iloc[:, -1]
    test_X,  test_y  = test_Xy.iloc[:, :-1],  test_Xy.iloc[:, -1]
    
    models = [SVR(), Ridge(), Lasso(), ElasticNet(), LinearRegression(), RandomForestRegressor(random_state=0), GradientBoostingRegressor(random_state=0)]
    df_baseline = Get_ML_Baseline(models, train_X, train_y, test_X, test_y)
    df_baseline_train = df_baseline[::2]
    df_baseline_test  = df_baseline[1::2]
    display(df_baseline_train)
    display(df_baseline_test)
    
    return df_baseline_train, df_baseline_test
    

#####################################################################################################
def Get_ML_Baseline(models, X_train, y_train, X_test, y_test):
    df_baseline_ = pd.DataFrame(columns=['model', 'explained_variance', 'MAE', 'MSE', 'RMSE', 'r2'])
    np.random.seed(42)
    for i, model in enumerate(models):
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        train_predicts = model.predict(X_train)
        train_report = regression_report(model_name+'_train', y_train, train_predicts, to_print=False)
        df_baseline_.loc[i*2] = train_report

        test_predicts  = model.predict(X_test)
        test_report = regression_report(model_name+'_test', y_test, test_predicts, to_print=False)
        df_baseline_.loc[i*2+1] = test_report
        
    return df_baseline_
