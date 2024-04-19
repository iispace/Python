# 테이블형 데이터셋의 이미지화 기법 
 
 
1. csv 파일 형태의 Dataset 읽어서 dataframe에 저장
   
3. Target column의 자료형에 따라 분류문제인지 회귀문제인지 확인하고, 자료형에 따라 dataframe 나누기
    2.1 num_df
    2.2 cat_df
        2.2.1 ord_df  : Ordinal Encoding 적용
        2.2.2 nom_df  
            2.2.2.1 분류 Dataset: Frequency Encoding
            2.2.2.2 회귀 Dataset: Target Encoding (Mean Encoding)

3. 범주형 데이터 => 수치형 데이터 변환 (Encoding Categorical Data to Numeric Data)
   
5. nom_df, ord_df, num_df 를 하나의 df로 concat한 후 train-test split => train_feat_df, test_feat_df
   
7. 이미지 변환시 각 채널에 적용하기 위해 데이터 정규화 또는 데이터 변환 후 수치형, 명목형, 순위형으로 분리
   
   5-1. For channel[0] : 데이터 정규화 후 수치형, 명목형, 순위형으로 분리 =>  train_encoded_sc_dfs, test_encoded_sc_dfs
   
         sc = MinMaxScaler()
         train_sc = sc.fit_transform(train_df),
         test_sc  = sc.transform(train_df)
   
   5-2. For channel[1] : 데이터 변환 후 수치형, 명목형, 순위형으로 분리 =>  train_encoded_power_tf_dfs, test_encoded_power_tf_dfs
   
         Yeo-Johnson transform 적용
   
   5-3. For channel[2] : 데이터 변환 후 수치형, 명목형, 순위형으로 분리 =>  train_encoded_log_tf_dfs, test_encoded_log_tf_dfs
   
         log transform 적용

9. 각 채널마다 train set과 test set에 대한 상관계수 행렬 생성
    
   6-1. For channel 0
   
        train_encoded_sc_corr_dict = Correlations(train_encoded_sc_dfs, train_df_label)
        test_encoded_sc_corr_dict  = Correlations(test_encoded_sc_dfs,  test_df_label)
  
   6-2. For channel 1
   
        train_encoded_power_tf_corr_dict = Correlations(train_encoded_power_tf_dfs, train_df_label)
        test_encoded_power_tf_corr_dict  = Correlations(test_encoded_power_tf_dfs,  test_df_label)
  
   6-3. For channel 2
   
        train_encoded_log_tf_corr_dict = Correlations(train_encoded_log_tf_dfs, train_df_label)
        test_encoded_log_tf_corr_dict  = Correlations(test_encoded_log_tf_dfs,  test_df_label)

11. 이미지 변환
    
  7-1. train set 채널별 이미지 행렬 생성
  
        train_ch0_converter = Image_converter(train_encoded_sc_dfs)
        train_ch0_sample_list, train_ch0_img_list   = train_ch0_converter.Generate_Channel_df(train_encoded_sc_corr_dict, max_value=255)
        train_ch0_images = train_ch0_converter.imagification(train_ch0_img_list)
        
        train_ch1_converter = Image_converter(train_encoded_power_tf_dfs)
        train_ch1_sample_list, train_ch1_img_list = train_ch1_converter.Generate_Channel_df(train_encoded_power_tf_corr_dict, max_value=255)
        train_ch1_images = train_ch1_converter.imagification(train_ch1_img_list)
        
        train_ch2_converter = Image_converter(train_encoded_log_tf_dfs)
        train_ch2_sample_list, train_ch2_img_list = train_ch2_converter.Generate_Channel_df(train_encoded_log_tf_corr_dict, max_value=255)
        train_ch2_images = train_ch2_converter.imagification(train_ch2_img_list)
    
  7-2. test set  채널별 이미지 행렬 생성
  
        test_ch0_converter = Image_converter(test_encoded_sc_dfs)
        test_ch0_sample_list, test_ch0_img_list   = test_ch0_converter.Generate_Channel_df(test_encoded_sc_corr_dict, max_value=255)
        test_ch0_images = test_ch0_converter.imagification(test_ch0_img_list)
        
        test_ch1_converter = Image_converter(test_encoded_power_tf_dfs)
        test_ch1_sample_list, test_ch1_img_list   = test_ch1_converter.Generate_Channel_df(test_encoded_power_tf_corr_dict, max_value=255)
        test_ch1_images = test_ch1_converter.imagification(test_ch1_img_list)
        
        test_ch2_converter = Image_converter(test_encoded_log_tf_dfs)
        test_ch2_sample_list, test_ch2_img_list   = test_ch2_converter.Generate_Channel_df(test_encoded_log_tf_corr_dict, max_value=255)
        test_ch2_images = test_ch2_converter.imagification(test_ch2_img_list)

8. 3D 이미지 변환
   
    train_chns = [train_ch0_images, train_ch1_images, train_ch2_images]
    train_3d_images = To3DImage(train_chns, padding=1)
     
    test_chns = [test_ch0_images, test_ch1_images, test_ch2_images]
    test_3d_images =  To3DImage(test_chns, padding=1) 
  
10. 3D 이미지로 변환된 샘플 시각화
    
  ![image](https://github.com/iispace/Python/assets/24539773/d3ffb32c-5bce-4e8f-9760-a5849a366da0)

 



</pre>
