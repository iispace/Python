from utils import mcm_utils 
  
# data (as pandas dataframes) 
X , y, feat_dtype_dict, feature_vars, target_var, enc = mcm_utils.read_uci_dataset("Support2")

y_state = mcm_utils.target_state(y)

######################################
##### Phase_1: Preprocessing #####
######################################
"""
  1.1 train_test_split
  1.2 group by dtypes
  1.3 numeric encoding
"""
from typing import Any
from sklearn.model_selection import train_test_split   
from utils.mcm_preprocessing import dtype_divider

print(f"nom_feat: {feat_dtype_dict['nom_feat']}")
print(f"ord_feat: {feat_dtype_dict['ord_feat']}")
print(f"num_feat: {len(feat_dtype_dict['num_feat'])}{feat_dtype_dict['num_feat']}")
          
test_size = 0.2
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y[target_var], random_state=42)         

train_data_dict = {'df_feature':train_X, 'df_target':train_y, 'nominal': feat_dtype_dict['nom_feat'], 'ordinal':feat_dtype_dict['ord_feat'], 'numeric': feat_dtype_dict['num_feat'], 'target_name': target_var}
test_data_dict  = {'df_feature':test_X,  'df_target':test_y,  'nominal': feat_dtype_dict['nom_feat'], 'ordinal':feat_dtype_dict['ord_feat'], 'numeric': feat_dtype_dict['num_feat'], 'target_name': target_var}

print("\n======== trainset =========")
train_dtype_groups = dtype_divider(train_data_dict)
print("\n========= testset ==========")
test_dtype_groups  = dtype_divider(test_data_dict)


from utils.mcm_preprocessing import NumericEncoder

# train-set, test-set의 순위형과 명목형 변수들을 수치형으로 변환하여 수치형 셋에 저장
train_cat_dict = {'train': {'ord_df': train_dtype_groups.ord_df, 'nom_df': train_dtype_groups.nom_df, 'target_df': train_dtype_groups.target_df}}
test_cat_dict  = {'test' : {'ord_df': test_dtype_groups.ord_df,  'nom_df': test_dtype_groups.nom_df,  'target_df': test_dtype_groups.target_df}}

ne = NumericEncoder(train=train_cat_dict, test=test_cat_dict)

# 원래부터 수치형이었던 변수들은 수치형변환이 필요치 않으므로 그대로 수치형 셋에 추가
encoded_set = ne.encoded_dict.copy()
encoded_set['train'].update({'num_df': train_dtype_groups.num_df})
encoded_set['test'].update({'num_df': test_dtype_groups.num_df})

for set_type in encoded_set.keys():
    for dtype_name in encoded_set[set_type].keys():
        print(f"[{set_type}]: {encoded_set[set_type][dtype_name].name}{encoded_set[set_type][dtype_name].shape}")
    print()


######################################
##### Phase_2: Matrix Generation #####
######################################
from utils.mcm_matrix_generation import matrixifier

train_m = matrixifier(ord_df=encoded_set['train']['ord_df'], nom_df=encoded_set['train']['nom_df'], num_df=encoded_set['train']['num_df'])
test_m  = matrixifier(ord_df=encoded_set['test']['ord_df'], nom_df=encoded_set['test']['nom_df'], num_df=encoded_set['test']['num_df'])

print(f"train_m.square_matrix.shape: {train_m.square_matrix.shape}")
print(f"test_m.square_matrix.shape : {test_m.square_matrix.shape}")

# 상관관계 행렬 (하삼각 수정하기 전!)
display(train_m.sub_corr_dfs_[0])

# 하삼각성분이 수정된 것
display(train_m.sub_corr_dfs_updated_[0])


#######################################
##### Phase_3: Channel Assignment #####
#######################################
from utils.scenario_2.mcm_channel_assign import channel_assignment 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

mcm = channel_assignment(encoded_set, [train_m.square_matrix, test_m.square_matrix], [train_y, test_y])


###########################################
##### Save converted data into a file #####
###########################################
import pickle, os
from utils import mcm_utils
 
EXP_DATA_DIR = rf".\MCMData"  # 데이터변환 없이 Numeric Coding만 한 결과를 3채널에 복사해서 만든 데이터
mcm_utils.create_folder(EXP_DATA_DIR, recreate=False)

Support2_Data_Dir = rf"{EXP_DATA_DIR}\Support2"
mcm_utils.create_folder(Support2_Data_Dir, recreate=True)

file_path = os.path.join(Support2_Data_Dir, "mcm_converted.pkl")

with open(file_path, 'wb') as file:
    pickle.dump(mcm, file)


###########################################
##### Load converted data from a file #####
###########################################
import numpy as np
import pandas as pd
import pickle, os
import torch
from utils import mcm_utils
from utils.mcm_channel_assign import channel_assignment

EXP_DATA_DIR = rf".\MCMtData"   
Support2_Data_Dir = rf"{EXP_DATA_DIR}\Support2"
file_path = os.path.join(Support2_Data_Dir, "mcm_converted.pkl")

# load mcm_converted.pkl
with open(file_path, 'rb') as file:
    loaded_mcm = pickle.load(file)
    
train_mcm = torch.tensor(loaded_ca.train_matrices)
train_targets = loaded_mcm.train_targets

test_mcm  = torch.tensor(loaded_ca.test_matrices)
test_targets = loaded_mcm.test_targets

C, H, W = train_mcm.size(1), train_mcm.size(2), train_mcm.size(3)
train_N, test_N = train_mcm.size(0), test_mcm.size(0)

train_indices = loaded_mcm.train_indices 
test_indices  = loaded_mcm.test_indices 

print(f"train_mcm: {train_mcm.size()}")
print(f"test_mcm : {test_mcm.size()}\n")
print(f"{train_N}, {C}, {H}, {W}")
print(f"{test_N}, {C}, {H}, {W}\n")
print(f"train_indicies: {train_indices[:5]}, ...")
print(f"test_indicies : {test_indices[:5]}, ...")
