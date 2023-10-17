from JJ_train_logger import load_model_architecture

feature_n = X_train_ts.shape[1]
output_n = y_train_ts.shape[1] 

# 모델 구조 설계
#input, h1, h2, h3, h4, h5 = 512, 256, 128, 64, 32, 16
#hidden_layers = [(input, nn.LeakyReLU()), (h1, nn.LeakyReLU()), (h2, nn.LeakyReLU()), 
#                 (h3, nn.LeakyReLU()), (h4, nn.LeakyReLU()), (h5, nn.LeakyReLU()), (output_n, None)]
#optimizer_config = {'class': optim.Adam, 'learning_rate': 0.0001, 'weight_decay': 0.1}

archt_file_name = r'.\MLP_architecture_2.json'
hidden_layers, criterion, optimizer_config = load_model_architecture(archt_file_name)
architecture = []
architecture.append((feature_n, nn.LeakyReLU()))
architecture.extend(hidden_layers)
architecture.append((output_n, None))

# 모델 객체 생성
torch.manual_seed(42) # 모델 객체가 만들어질 때마다 같은 초기 가충치가 생성되도록 하기 위함. 
#model = MLP(feature_n, hidden_layers, optimizer_config, dropout_p=0.1) 
model = MLP(feature_n, architecture, optimizer_config, dropout_p=0.1) 

print(model)
#for param in model.parameters():
#  print(f'{param.shape}{param.dtype}\t{param}')
