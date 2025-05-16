# Generator, Discrminator 모델 생성
from models.Pix2Pix_RGB2NIR import NIR_GeneratorUNet, Discriminator, initialize_weights

model_gen = NIR_GeneratorUNet(in_channels=3, out_channels=3).to(device)
model_dis = Discriminator(in_channels=3).to(device)


# 가중치 초기화 적용
model_gen.apply(initialize_weights);
model_dis.apply(initialize_weights);




experiment_no = 4
