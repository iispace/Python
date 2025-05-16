import json
from datetime import datetime

from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt

import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from Models import RGB_NIR_Dataset
from Models.RGB_NIR_Dataset import dataloader_info, dataset_info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 실험 번호 부여
experiment_no = 4   # <= argument로 전달할 값


# ds, dl 생성
train_img_dir = r".\Datasets\RGB2NIR\train"
test_img_dir  = r".\Datasets\RGB2NIR\test"

train_ds = RGB_NIR_Dataset(train_img_dir, transform=transform_basic)
test_ds  = RGB_NIR_Dataset(test_img_dir,  transform=transform_basic)

print('\n'.join([f"{k} {v}" for k, v in dataset_info(train_ds).items()]), "\n", "-"*100)
print('\n'.join([f"{k} {v}" for k, v in dataset_info(test_ds).items()]))

batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

print("\n".join([f"{k} {v}" for k, v in dataloader_info(train_dl).items()]))

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

# 학습
model_gen.train()
model_dis.train()

batch_count = 0
num_epochs = 100
start_time = time.time()

loss_hist = {'gen':[],
             'dis':[]}

# 적용할 color_space 선택  # <==== argument로 받아서 처리할 수 있도록 수정할 부분
#color_space_transformer = None
#color_space_transformer = RGB2LAB_Transform()
#color_space_transformer = RGB2LCH_Transform()  
#color_space_transformer = RGB2YCbCr_Transform("2020")
color_space_transformer = RGB2YUV_Transform()  
#color_space_transformer = RGB2XYZ_Transform()  


for epoch in range(num_epochs):
    for rgb, nir, _ in train_dl:  # "RGB2NIR"이므로 rgb, nir 순서로 이미지 가져옴. # torch.Size([32, 3, 256, 256])
        batch_size = nir.size(0)
                    
        # real image
        real_rgb_ = rgb.to(device)  # RGB

        if color_space_transformer is None:
            input_img = real_rgb_
        else:
            input_img = color_space_transformer(real_rgb_)

        real_nir = nir.to(device)  # NIR

        # patch label
        real_label = torch.ones(batch_size, *patch, requires_grad=False).to(device)   # torch.Size([32, 1, 16, 16]) <= [B, 1, P, P]
        fake_label = torch.zeros(batch_size, *patch, requires_grad=False).to(device)  # torch.Size([32, 1, 16, 16]) <= [B, 1, P, P]
        
        ########## generator ##########
        model_gen.zero_grad()

        fake_nir = model_gen(input_img) # 실제 RGB 사진을 입력 받아서 가짜 NIR 이미지 생성, feature extraction을 위한 UNetDown으로 들어가므로 일반적인 GAN처럼 random noize를 주지 않고 바로 real image를 입력으로 줌.
        #fake_nir = std_ * fake_nir + mean_  ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< try result: Bed
      
        # 가짜 이미지에 대한 판별
        out_dis = model_dis(input_img, fake_nir) # 실제 입력된 이미지(real RGB)를 condition으로 주고, 생성된 가짜 NIR 이미지가 가짜인지 진짜인지를 패치 단위로 예측, # torch.Size([32, 1, 16, 16]) <= [B, 1, P, P]
        
            
        gen_loss = loss_func_gan(out_dis, real_label)  # nn.BCELoss()
        # pixel-wise loss
        pixel_loss = loss_func_pix(fake_nir, real_nir) # nn.L1Loss()

        g_loss = gen_loss + (lambda_pixel * pixel_loss)
        g_loss.backward()
        opt_gen.step()

        ########## discriminator ##########
        model_dis.zero_grad()

        # 판별자에게 진짜 이미지와 가짜 이미지를 5:5로 주어 판별자가 학습할 수 있도록 함.
        # 진짜 이미지에 대한 판별
        out_dis = model_dis(input_img, real_nir)   # condition과 실제 이미지를 주어서 Discriminator가 각 이미지별 패치 단위로 판별하도록 함함
        real_loss = loss_func_gan(out_dis, real_label)
        
        # 가짜 이미지에 대한 판별
        out_dis = model_dis(real_nir, fake_nir.detach())  
        #out_dis = model_dis(input_img, fake_nir.detach()) # <<<<<<<<<<<<<<<< try result : bed  
        fake_loss = loss_func_gan(out_dis, fake_label)      # nn.BCELoss()

        d_loss = (real_loss + fake_loss) * 0.5
        d_loss.backward()
        opt_dis.step()

        loss_hist['gen'].append(g_loss.item())
        loss_hist['dis'].append(d_loss.item())

        batch_count += 1
        if batch_count % 100 == 0:
            print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))
            
print(f"\ncolor_space_transformer: {color_space_transformer.__class__.__name__}")


