# 테이블형 데이터셋의 이미지화 기법 
 
### 데이터 로드

1. csv 파일 형태의 Dataset 읽어서 dataframe에 저장

### 데이터 전처리
   
2. Target column의 자료형에 따라 분류문제인지 회귀문제인지 확인하고, 자료형에 따라 dataframe 나누기<br>
    nom_cols = [...]<br>
    ord_cols = [...]<br>
    num_cols = [col for col in df.columns.tolist() if col not in nom_cols and col not in ord_cols]<br>
    2.1 num_df = df[num_cols] # (수치형)<br>
    2.2 ord_df = df[ord_cols] # (범주형 > 순위형)<br>
    2.3 nom_df = df[nom_cols] # (범주형 > 명목형)<br>
   
       
3. 범주형 데이터 => 수치형 데이터 변환 (Encoding Categorical Data to Numeric Data)
    3.1 ord_df  : Ordinal Encoding <br>
    3.2 nom_df  <br>
       3.2.1 분류 Dataset: Frequency Encoding<br>
       3.2.2 회귀 Dataset: Target Encoding (Mean Encoding)<br>
   
4. 수치형으로 변환된 nom_df, ord_df, num_df 를 하나의 df로 concat한 후 train-test split => train_X_df, test_X_df

### 이미지화 준비
   
5. 3채널 이미지화를 위한 채널별 행렬 생성 
   
   5-1. For channel[0] : 데이터 정규화 및 데이터 유형별 재분리
   
         # 정규화
   
         feat_cols = train_X_df.columns().tolist()
         sc = MinMaxScaler()
         train_sc_df = pd.DataFrame(sc.fit_transform(train_X_df), columns=feat_cols)
         test_sc_df  = pd.DataFrame(sc.transform(train_X_df), columns=feat_cols)
   
         # 데이터 유형별 재분리
         train_X_dfs_CH0 = [train_nom_df_CH0, train_ord_df_CH0, train_nom_df_CH0]     
         test_X_dfs_CH0  = [test_nom_df_CH0, test_ord_df_CH0, test_nom_df_CH0]     
   
   5-2. For channel[1] : 데이터 변환(Yeo-Johnson Transform) 후 데이터 유형별 재분리 
   
         Yeo-Johnson transform 적용
   
         train_X_dfs_CH1 = [train_nom_df_CH1, train_ord_df_CH1, train_nom_df_CH1]
         test_X_dfs_CH1  = [test_nom_df_CH1, test_ord_df_CH1, test_nom_df_CH1]     
      
     5-3. For channel[2] : 데이터 변환 후 수치형, 명목형, 순위형으로 분리
   
         log transform 적용

         train_X_dfs_CH2 = [train_nom_df_CH2, train_ord_df_CH2, train_nom_df_CH2]     
         test_X_dfs_CH2 = [test_nom_df_CH2, test_ord_df_CH2, test_nom_df_CH2]    

6. 각 채널마다 train set과 test set에 대한 상관계수 행렬 생성
    
   6-1. For channel 0
   
        train_corr_dict_CH0 = Correlations(train_X_dfs_CH0, train_df_label)
        test_corr_dict_CH0  = Correlations(test_X_dfs_CH0,  test_df_label)
  
   6-2. For channel 1
   
        train_corr_dict_CH1 = Correlations(train_X_dfs_CH1, train_df_label)
        test_corr_dict_CH1  = Correlations(test_X_dfs_CH1,  test_df_label)
  
   6-3. For channel 2
   
        train_corr_dict_CH2 = Correlations(train_X_dfs_CH2, train_df_label)
        test_corr_dict_CH2  = Correlations(test_X_dfs_CH2,  test_df_label)
   

### 이미지 변환
    
  7-1. train set 채널별 이미지 행렬 생성
  
        train_ch0_converter = Image_converter(train_X_dfs_CH0)
        train_ch0_sample_list, train_ch0_img_list   = train_ch0_converter.Generate_Channel_df(train_corr_dict_CH0, max_value=255)
        train_ch0_images = train_ch0_converter.imagification(train_ch0_img_list)
        
        train_ch1_converter = Image_converter(train_X_dfs_CH1)
        train_ch1_sample_list, train_ch1_img_list = train_ch1_converter.Generate_Channel_df(train_corr_dict_CH1, max_value=255)
        train_ch1_images = train_ch1_converter.imagification(train_ch1_img_list)
        
        train_ch2_converter = Image_converter(train_X_dfs_CH2)
        train_ch2_sample_list, train_ch2_img_list = train_ch2_converter.Generate_Channel_df(train_corr_dict_CH2, max_value=255)
        train_ch2_images = train_ch2_converter.imagification(train_ch2_img_list)
    
  7-2. test set  채널별 이미지 행렬 생성
  
        test_ch0_converter = Image_converter(test_X_dfs_CH0)
        test_ch0_sample_list, test_ch0_img_list   = test_ch0_converter.Generate_Channel_df(test_corr_dict_CH0, max_value=255)
        test_ch0_images = test_ch0_converter.imagification(test_ch0_img_list)
        
        test_ch1_converter = Image_converter(test_X_dfs_CH1)
        test_ch1_sample_list, test_ch1_img_list   = test_ch1_converter.Generate_Channel_df(test_corr_dict_CH1, max_value=255)
        test_ch1_images = test_ch1_converter.imagification(test_ch1_img_list)
        
        test_ch2_converter = Image_converter(test_X_dfs_CH2)
        test_ch2_sample_list, test_ch2_img_list   = test_ch2_converter.Generate_Channel_df(test_corr_dict_CH2, max_value=255)
        test_ch2_images = test_ch2_converter.imagification(test_ch2_img_list)

8. 3D 이미지 변환
   
    train_chns = [train_ch0_images, train_ch1_images, train_ch2_images]
    train_3d_images = To3DImage(train_chns, padding=1)
     
    test_chns = [test_ch0_images, test_ch1_images, test_ch2_images]
    test_3d_images =  To3DImage(test_chns, padding=1) 
  
9. 3D 이미지로 변환된 샘플 시각화
    
  ![image](https://github.com/iispace/Python/assets/24539773/d3ffb32c-5bce-4e8f-9760-a5849a366da0)

 



</pre>
