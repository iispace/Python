import shutil, os

def create_folder(dir, recreate=True):
    if os.path.exists(dir):
        if recreate:
            shutil.rmtree(dir)
        else:
            print(f"The folder {dir} already exists !!")
            return
    os.makedirs(dir)
    
######################################################################################
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OrdinalEncoder 
import pandas as pd 
import numpy as np 

def read_uci_dataset(dataset_name):    
    if dataset_name.lower() == "support2":
        # fetch dataset 
        data = fetch_ucirepo(id=880) 
        
        # data (as pandas dataframes) 
        X = data.data.features 
        ys = data.data.targets 
        
        X = X.drop(columns=['dementia'])
        
        feature_vars = X.columns.tolist()
        respons_vars = ys.columns.tolist()
        target_var = 'sfdm2'
        
        # remove all samples having at least one missing value in any feature
        Xy_ = pd.concat([X, ys], axis=1)
        Xy = Xy_.dropna()
        sample_idxs = Xy.index.tolist()
        
        X  = Xy[feature_vars]
        ys = Xy[respons_vars]
        
        # numeric encoding of the target variable
        enc = OrdinalEncoder() 
        
        y = ys[target_var].to_frame()
        enc.fit(y)
        y = pd.DataFrame(enc.transform(y), columns=[target_var], index=sample_idxs)
        
        
        num_feat = X.select_dtypes(include=['number']).columns.tolist()
        cat_feat = X.select_dtypes(exclude=['number']).columns.tolist()
        ord_feat = ['num.co', 'income']
        nom_feat = [col for col in cat_feat if col not in ord_feat]
        num_feat = [col for col in num_feat if col not in ord_feat and col not in nom_feat]  
        
        feat_dtype_dict = {'num_feat': num_feat, 'nom_feat': nom_feat, 'ord_feat': ord_feat}
        
        print(f"\nX.shape: {X.shape}, y.shape: {y.shape}")
        
        
        return X, y, feat_dtype_dict, feature_vars, target_var, enc
    
    
    
######################################################################################
def target_state(y_):
    state = y_.value_counts().to_frame()
    state['ratio'] = state['count'] / state['count'].sum() * 100.
    state['ratio'] = state['ratio'].map('{:,.2f}%'.format)
    state.sort_index(inplace=True)
    display(state)
  
    return state


#######################################################################################
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(features_df):
    numeric_sub_df = features_df.select_dtypes(include=['float', 'int'])

    vif_df = pd.DataFrame()
    vif_df['Feature'] = numeric_sub_df.columns
    vif_df['VIF'] = [variance_inflation_factor(numeric_sub_df.values, i) for i in range(numeric_sub_df.shape[1])]
    
    # 가로로 두 번 반복되는 표로 출력하기 위해 수치형 변수의 개수인 34를 2로 나눈 17행을 갖는 두 개의 df로 생성
    vif_df1 = vif_df.iloc[:17, :]
    vif_df1.reset_index(inplace=True)

    vif_df2 = vif_df.iloc[17:17*2, :]
    vif_df2.reset_index(inplace=True)

    vif_df_12 = pd.concat([vif_df1, vif_df2], axis=1)
    display(vif_df_12)
    
    return vif_df_12
