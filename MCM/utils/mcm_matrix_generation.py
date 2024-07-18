# 행렬 생성 단계에서 사용되는 함수(하삼각성분의 값을 상삼각성분 값의 제곱으로 수정하는 함수)
import numpy as np 
def update_trilow(corr_df_):
    N = corr_df_.shape[0]
    corr_triu = np.triu(corr_df_.values)
    # 하삼각성분을 상삼각성분의 제곱으로 수정 
    for i in range(N):
        for j in range(i):
            corr_triu[i, j] = corr_triu[j, i] ** 2
    return corr_triu
    
################################################################################################################
import pandas as pd 

class matrixifier():
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        
        self.ord_corr_df = None  # 자료형별 상관계수 행렬(순위형)
        self.nom_corr_df = None  # 자료형별 상관계수 행렬(명목형)
        self.num_corr_df = None  # 자료형별 상관계수 행렬(수치형)
        
        self.ord_corr_updated_df = None # 자료형별 상관계수 행렬(순위형)을 이용하여 하삼각성분 값을 수정한 행렬 
        self.nom_corr_updated_df = None # 자료형별 상관계수 행렬(명목형)을 이용하여 하삼각성분 값을 수정한 행렬 
        self.num_corr_updated_df = None # 자료형별 상관계수 행렬(수치형)을 이용하여 하삼각성분 값을 수정한 행렬 
        
        self.sub_corr_dfs_ = []          # 자료형별 상관계수 행렬들의 리스트 
        self.sub_corr_dfs_updated_ = []  # 하삼각성분의 값이 상삼각성분 값의 제곱으로 업데이트 된 상태
        
        self.N = 0
        self.square_matrix = None
        self.feature_vars = []
        
        self.__call__()  # 상관계수 행렬 생성 후 상삼각성분 값의 제곱으로 하삼각성분값 수정
        
    def __call__(self):
        keys = self.kwargs.keys()
         
        for key in keys:
            if 'num' in key:
                self.num_corr_df = pd.DataFrame(self.kwargs[key].corr())
                self.num_corr_df.name = "num"
                self.sub_corr_dfs_.append(self.num_corr_df)
                self.num_corr_updated_df = pd.DataFrame(update_trilow(self.num_corr_df), columns=self.num_corr_df.columns, index=self.num_corr_df.index)
                self.feature_vars.extend(self.num_corr_updated_df.columns.tolist())
                self.num_corr_updated_df.name = "num_updated"
                self.sub_corr_dfs_updated_.append(self.num_corr_updated_df)
                self.N += self.num_corr_updated_df.shape[0]
                
            elif 'nom' in key:
                self.nom_corr_df = self.kwargs[key].corr()
                self.nom_corr_df.name = "nom_df"
                self.sub_corr_dfs_.append(self.nom_corr_df)
                self.nom_corr_updated_df = pd.DataFrame(update_trilow(self.nom_corr_df), columns=self.nom_corr_df.columns, index=self.nom_corr_df.index)
                self.feature_vars.extend(self.nom_corr_updated_df.columns.tolist())
                self.nom_corr_updated_df.name = "nom_updated"
                self.sub_corr_dfs_updated_.append(self.nom_corr_updated_df)
                self.N += self.nom_corr_updated_df.shape[0]
            
            elif 'ord' in key:
                self.ord_corr_df = self.kwargs[key].corr()
                self.ord_corr_df.name = "ord_df"
                self.sub_corr_dfs_.append(self.ord_corr_df)
                self.ord_corr_updated_df = pd.DataFrame(update_trilow(self.ord_corr_df), columns=self.ord_corr_df.columns, index=self.ord_corr_df.index)
                self.feature_vars.extend(self.ord_corr_updated_df.columns.tolist())
                self.ord_corr_updated_df.name = "ord_updated"
                self.sub_corr_dfs_updated_.append(self.ord_corr_updated_df)
                self.N += self.ord_corr_updated_df.shape[0]

        # 자료형별 상관계수 행렬(하삼각 수정된 행렬)을 하나로 합치기    
        square_matrix = np.full((self.N, self.N), 0.0)
        start = 0
        end = 0
        for df_ in self.sub_corr_dfs_updated_:
            end += df_.shape[0]
            square_matrix[start:end, start:end] = df_
            start = end

        self.square_matrix = pd.DataFrame(square_matrix, columns=self.feature_vars, index=self.feature_vars)
