# Generator, Discrminator 모델 생성
from models.Pix2Pix_RGB2NIR import *

model_gen = NIR_GeneratorUNet(in_channels=3, out_channels=3).to(device)
model_dis = Discriminator(in_channels=3).to(device)








experiment_no = 4
