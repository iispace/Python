# 테이블형 데이터셋의 3채널 행렬화 기법 
 
### 0. 데이터 로드

csv 파일 형태의 Dataset 읽어서 dataframe에 저장

### 1. 데이터 전처리

1. train-test split    
2. Target column의 자료형에 따라 분류문제인지 회귀문제인지 확인하고, 자료형에 따라 dataframe 나누기<br>
    nom_cols = [...]<br>
    ord_cols = [...]<br>
    num_cols = [col for col in df.columns.tolist() if col not in nom_cols and col not in ord_cols]<br>
    2.1 num_df = df[num_cols] # (수치형)<br>
    2.2 ord_df = df[ord_cols] # (범주형 > 순위형)<br>
    2.3 nom_df = df[nom_cols] # (범주형 > 명목형)<br>
       
3. 범주형 데이터 => 수치형 데이터 변환 (Encoding Categorical Data to Numeric Data)<br>
    <pre>
    3.1 ord_df  : Ordinal Encoding 
    3.2 nom_df  
       3.2.1 분류 Dataset: Frequency Encoding
       3.2.2 회귀 Dataset: Target Encoding (Mean Encoding)
    </pre>
### 2. 행렬 생성 단계
   
   3채널 행렬화를 위한 채널별 행렬 생성 
<pre>
   2-1. For channel[0] : 데이터 유형별 상관계수 행렬 생성 및 데이터 정규화
        # 정규화 : min-max normalization tabular data<br>
        
        # 데이터 유형별 상관계수 행렬 생성 & concat<br>
           - nom_df_encoded: Kendal correlation coefficient        ==> nom_corr
           - ord_df_encoded: Spearman Rank correlation coefficient ==> ord_corr
           - num_df        : Pearson correlation coefficient       ==> num_corr<br>
        
         # 상관계수 행렬의 대각성분  : 정규화된 특성벡터의 값 일대일 할당
         # 상관계수 행렬의 하삼각성분: 상삼각성분 값의 제곱 할당
 
         => CH0_matrix = concat(nom_corr, ord_corr, num_corr)
</pre>  
<pre>
   2-2. For channel[1] :  
         # 정규화 :  Yeo-Johnson transform (for more Gaussian-like distribution) <br>
         # 상관계수 행렬 생성: Channel[0]과 동일한 방법
 
         => CH1_matrix = concat(nom_corr, ord_corr, num_corr)
</pre> 
<pre>
   2-3. For channel[2] : 
          # 정규화 :  Quantile transform (Robust to outliers) <br>
         # 상관계수 행렬 생성: Channel[0]과 동일한 방법
  
         => CH2_matrix = concat(nom_corr, ord_corr, num_corr)
</pre>
 
### 3. 3채널 행렬화(이미지 구조화) 단계
    
      M_list = [CH0_matrix, CH1_matrix, CH2_matrix]
      3D_Matrix = torch.cat(M_list, dim=0).reshape(-1,img_size, img_size)
 
### 4. 3채널 행렬화(이미지 구조화) 결과물 시각화 예시
    
  ![image](https://github.com/iispace/Python/assets/24539773/d3ffb32c-5bce-4e8f-9760-a5849a366da0)

 




