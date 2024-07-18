import os
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


#######################################################
# 기계학습용 데이터셋 (테이블형 데이터셋 수치화 된 것) 저장
#######################################################

EXP_DATA_DIR = rf".\Data\"  # 데이터변환 없이 Numeric Coding만 한 결과를 3채널에 복사해서 만든 데이터
mcm_utils.create_folder(EXP_DATA_DIR, recreate=False)

ML_train_Xy = pd.concat([encoded_set['train']['num_df'], encoded_set['train']['nom_df'], encoded_set['train']['ord_df'], train_y], axis=1)
ML_test_Xy  = pd.concat([encoded_set['test']['num_df'],  encoded_set['test']['nom_df'],  encoded_set['test']['ord_df'],  test_y], axis=1)

print(f"ML_train: {ML_train_Xy.shape}")
print(f"ML_test : {ML_test_Xy.shape}")

ML_Data_Dir = rf"{EXP_DATA_DIR}\ML"
mcm_utils.create_folder(ML_Data_Dir, recreate=True)

ML_train_Xy.to_csv(os.path.join(ML_Data_Dir, "ML_train.csv"), index=True)
ML_test_Xy.to_csv(os.path.join(ML_Data_Dir, "ML_test.csv"), index=True)
