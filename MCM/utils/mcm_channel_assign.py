import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer

class channel_assignment():   # 채널별로 데이터변환 적용한 변환 기법
    def __init__(self, encoded_set, square_matrices, targets):
        """ 
            encoded_set: {'train': {'ord_df': ord_df, 'nom_df': nom_df, 'num_df': num_df},
                          'test' : {'ord_df': ord_df, 'nom_df': nom_df, 'num_df': num_df}}
            square_matrices: [train_m.square_matrix, test_m.square_matrix]
            targets        : [train_y: pd.DataFrame, test_y: pd.DataFrame]
        """
        self.encoded_set = encoded_set
        self.square_matrices = square_matrices
        self.train_square_matrix = square_matrices[0]
        self.test_square_matrix = square_matrices[1]
        self.train_targets = targets[0]
        self.test_targets = targets[1]
        
        self.train_bases = self.__bases__(encoded_set, "train")  # 대각성분에 수치형으로 인코딩된 특성변수값 채워서 기본 행렬 완성
        self.train_indices = [df_.name for df_ in self.train_bases]
        
        self.test_bases  = self.__bases__(encoded_set, "test")   # 대각성분에 수치형으로 인코딩된 특성변수값 채워서 기본 행렬 완성
        self.test_indices = [df_.name for df_ in self.test_bases]
        
        self.train_ch0s, self.test_ch0s = self.__ch0s__()
        self.train_ch1s, self.test_ch1s = self.__ch1s__()
        self.train_ch2s, self.test_ch2s = self.__ch2s__()
        
        self.train_matrices = np.stack([self.train_ch0s, self.train_ch1s, self.train_ch2s], axis=1)
        self.test_matrices  = np.stack([self.test_ch0s,  self.test_ch1s,  self.test_ch2s],  axis=1)
        
    def __bases__(self, dtype_subsets, key):
        X_encoded = pd.concat([self.encoded_set[key]['num_df'], self.encoded_set[key]['nom_df'], self.encoded_set[key]['ord_df']], axis=1)
        X_encoded = X_encoded[self.train_square_matrix.columns]
        square_matrix = self.train_square_matrix if key == 'train' else self.test_square_matrix
        
        if key == "train":
            self.train_X_encoded = X_encoded.copy()  # 수치형 인코딩된 테이블형 데이터셋 (trainset)
        else:
            self.test_X_encoded = X_encoded.copy()   # 수치형 인코딩된 테이블형 데이터셋 (testset)
        
        samples = []
        # 샘플 하나에 해당하는 행렬마다 대각성분에 특성변수값 할당하여 기본 행렬 생성
        for index in X_encoded.index.tolist():
            sample = square_matrix.copy() # pandas.DataFrame(41, 41)
            sample.name = index
            for i in range(sample.shape[0]):
                value = X_encoded.loc[index].iloc[i]
                sample.iloc[i, i] = value
            samples.append(sample)
       
        return samples   # list of pd.DataFrame

    
    def __ch0s__(self):
        # min_max normalization
        sc = MinMaxScaler()
        sc.fit(self.train_X_encoded)
        train_X_encoded_sc = pd.DataFrame(sc.transform(self.train_X_encoded), columns=self.train_X_encoded.columns, index=self.train_X_encoded.index)
        test_X_encoded_sc  = pd.DataFrame(sc.transform(self.test_X_encoded), columns=self.test_X_encoded.columns, index=self.test_X_encoded.index)

        train_ch0s, test_ch0s = self.__update_diagonal_elements__(train_X_encoded_sc, test_X_encoded_sc)
               
        return train_ch0s, test_ch0s
    
    def __ch1s__(self):
        # Yeo-Johnson transformation
        ptf = PowerTransformer('yeo-johnson', standardize=True)
        ptf.fit(self.train_X_encoded)
        train_X_encoded_ptf = pd.DataFrame(ptf.transform(self.train_X_encoded), columns=self.train_X_encoded.columns, index=self.train_X_encoded.index)
        test_X_encoded_ptf  = pd.DataFrame(ptf.transform(self.test_X_encoded), columns=self.test_X_encoded.columns, index=self.test_X_encoded.index)
        
        train_ch1s, test_ch1s = self.__update_diagonal_elements__(train_X_encoded_ptf, test_X_encoded_ptf)
                   
        return train_ch1s, test_ch1s
    
    def __ch2s__(self):
        # Quantile transformation
        qtf = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
        qtf.fit(self.train_X_encoded)
        train_X_encoded_qtf = pd.DataFrame(qtf.transform(self.train_X_encoded), columns=self.train_X_encoded.columns, index=self.train_X_encoded.index)
        test_X_encoded_qtf  = pd.DataFrame(qtf.transform(self.test_X_encoded), columns=self.test_X_encoded.columns, index=self.test_X_encoded.index)
        
        train_ch2s, test_ch2s = self.__update_diagonal_elements__(train_X_encoded_qtf, test_X_encoded_qtf)
                   
        return train_ch2s, test_ch2s
    
    def __update_diagonal_elements__(self, train_X_encoded_tf, test_X_encoded_tf):
        train_chs, test_chs = [], []
        # 샘플 하나에 해당하는 행렬마다 대각성분에 특성변수값 할당하여 기본 행렬 생성
        for index in self.train_X_encoded.index.tolist():
            sample = self.train_square_matrix.copy() # pandas.DataFrame(41, 41)
            sample.name = index
            for i in range(sample.shape[0]):
                value = train_X_encoded_tf.loc[index].iloc[i]
                sample.iloc[i, i] = value
            train_chs.append(sample)
        
        for index in self.test_X_encoded.index.tolist():
            sample = self.test_square_matrix.copy() # pandas.DataFrame(41, 41)
            sample.name = index
            for i in range(sample.shape[0]):
                value = test_X_encoded_tf.loc[index].iloc[i]
                sample.iloc[i, i] = value
            test_chs.append(sample)    
            
        return train_chs, test_chs
