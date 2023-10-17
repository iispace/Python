from JJ_train_logger import load_model_architecture
 
feature_n = X_train_ts.shape[1]
output_n = y_train_ts.shape[1] 

# 모델 구조 변수
architecture = []
# 모델 구조 변수에 입력층 추가
architecture.append((feature_n, nn.LeakyReLU()))

# 모델 구조 변수에 은닉층 추가
archt_file_name = r'.\MLP_architecture_2.json'
hidden_layers, criterion, optimizer_config = load_model_architecture(archt_file_name)
architecture.extend(hidden_layers)

# 모델 구조 변수에 출력층 추가
architecture.append((output_n, None))

# 모델 객체 생성
torch.manual_seed(42) # 모델 객체가 만들어질 때마다 같은 초기 가충치가 생성되도록 하기 위함. 
model = MLP(feature_n, architecture, optimizer_config, dropout_p=0.1) 

print(model)
#for param in model.parameters():
#  print(f'{param.shape}{param.dtype}\t{param}')
