# Generator, Discrminator 모델 생성
from models.Pix2Pix_RGB2NIR import NIR_GeneratorUNet, Discriminator, initialize_weights

model_gen = NIR_GeneratorUNet(in_channels=3, out_channels=3).to(device)
model_dis = Discriminator(in_channels=3).to(device)


# 가중치 초기화 적용
model_gen.apply(initialize_weights);
model_dis.apply(initialize_weights);

# 손실함수
loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# loss_func_pix 가중치
lambda_pixel = 100

# patch 수
patch = (1,256//2**4,256//2**4)
print(f"patch size: {patch}")

# 최적화 파라미터
from torch import optim
lr = 2e-4
beta1 = 0.5
beta2 = 0.999

opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))
#opt_dis = optim.SGD(model_dis.parameters(),lr=lr)
#opt_gen = optim.SGD(model_gen.parameters(),lr=lr)


experiment_no = 4
