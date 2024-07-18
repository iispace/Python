import pandas as pd 
import numpy as np 

class dtype_divider():
    def __init__(self, args_dict) -> None:
        """ 
        args_dict = {'df_feature':train_X(pandas.DataFrame), 'df_target':train_y(pandas.DataFrame), 
                     'nominal': nom_feat(list of string), 'ordinal':ord_feat(list of string), 'numeric': num_feat(list of string), 
                     'target_name': target_name(list of string)}
                    
        """
        super().__init__()
        self.args_dict = args_dict
        try:
            self.df_X = args_dict['df_feature']
            self.df_y = args_dict['df_target']
        except KeyError as ke:
            print(f"There is no such a key, {ke}")
            return
        except Exception as e:
            print(e)
            return
        
        self.ord_feat = None  
        self.nom_feat = None  
        self.num_feat = None  
        self.target_name = None
        
        self.ord_df = None 
        self.nom_df = None 
        self.num_df = None 
        self.target_df = None 
        
        self.__call1__() # group by data types
        self.__print__()
        
        
    def __call1__(self):
        keys = self.args_dict.keys()
        for key in keys:
            if key == 'nominal':
                self.nom_feat = self.args_dict[key]
                self.nom_df = self.df_X[self.nom_feat]
                self.nom_df.name = 'nom_df'
            elif key == 'ordinal':
                self.ord_feat = self.args_dict[key]
                self.ord_df = self.df_X[self.ord_feat]
                self.ord_df.name = 'ord_df'
            elif key == 'numeric':
                self.num_feat = self.args_dict[key]    
                self.num_df = self.df_X[self.num_feat]
                self.num_df.name = 'num_df'
            elif key == 'target_name':
                self.target_name = self.args_dict[key]
                self.target_df = self.df_y[self.target_name]
                
    def __print__(self):
        print(f"ord_df: {self.ord_df.shape}")
        print(f"num_df: {self.num_df.shape}")
        print(f"nom_df: {self.nom_df.shape}")
        print(f"target_df: {self.target_df.shape}")
 
##########################################################################################################################                
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
                
class NumericEncoder():
    def __init__(self, **kwargs):
        """ 
            args_dict: {'train': {'ord_df': ord_df, 'nom_df': nom_df, 'target_df': target_df}, 
                        'test' : {'ord_df': ord_df, 'nom_df': nom_df, 'target_df': target_df}}
            
        """
        super().__init__
        self.kwargs = kwargs
        self.train = kwargs['train']
        self.test = kwargs['test']
        self.encoded_dict = self.__call__()
        
    def __call__(self):
        outputs = {'train': {}, 'test': {}}
        for data_dicts in zip(self.train.items(), self.test.items()):
            train_dict = data_dicts[0][1]
            train_target = data_dicts[0][1]['target_df']
            test_dict = data_dicts[1][1]
            #test_target = data_dicts[1][1]['target_df']
            
            for train_dtype_name, test_dtype_name in zip(train_dict.keys(), test_dict.keys()):  # 'ord_df' or 'nom_df'
                if train_dtype_name == 'nom_df' and test_dtype_name == 'nom_df':
                    enc = TargetEncoder() 
                    enc.fit(train_dict[train_dtype_name], train_target)
                    train_encoded = enc.transform(train_dict[test_dtype_name])
                    train_encoded.name = 'nom_df'
                    test_encoded = enc.transform(test_dict[test_dtype_name] )
                    test_encoded.name = 'nom_df'
                    outputs['train'].update({'nom_df': train_encoded})
                    outputs['test'].update({'nom_df': test_encoded})
                    
                elif train_dtype_name == 'ord_df' and test_dtype_name == 'ord_df':
                    cols = test_dict[test_dtype_name].columns.tolist()
                    train_idxs = train_dict[train_dtype_name].index.tolist()
                    test_idxs = test_dict[test_dtype_name].index.tolist()
                    enc = OrdinalEncoder()
                    enc.fit(train_dict[train_dtype_name])
                    train_encoded = pd.DataFrame(enc.transform(train_dict[train_dtype_name]), columns=cols, index=train_idxs)
                    train_encoded.name = 'ord_df'
                    test_encoded  = pd.DataFrame(enc.transform(test_dict[test_dtype_name]),   columns=cols, index=test_idxs)
                    test_encoded.name = 'ord_df'
                    outputs['train'].update({'ord_df': train_encoded})
                    outputs['test'].update({'ord_df':test_encoded})
        
        return outputs                
