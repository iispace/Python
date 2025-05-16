
import json
from datetime import datetime

#from os import listdir
#from os.path import join
#import random
import matplotlib.pyplot as plt

import os
import time
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#from torchvision.transforms.functional import to_pil_image
from models.RGB_NIR_Dataset import RGB_NIR_Dataset
from models.RGB_NIR_Dataset import dataloader_info, dataset_info
from color_space.convert import *



transform_basic = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((256,256)),
])

transform_Norm = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.Resize((256,256)),
])

transform_to_YCbCr = transforms.Compose([
    # dataloader에서 가져올 때 이미 tensor로 변환되었으므로, 여기서는 색 공간 변환만 처리함.
    RGB2YCbCr_Transform(BT="2020"),  # 사용자 정의 변환기 추가
])

transform_to_Lab = transforms.Compose([
    RGB2LAB_Transform(),
])

transform_to_Yuv = transforms.Compose([
    RGB2YUV_Transform(),
])
transform_to_xyz = transforms.Compose([
    RGB2XYZ_Transform(),
])

transform_to_ycbcr_kornia = transforms.Compose([
    RGB2YCbCr_kornia_Transform(),
])

transform_to_gray = transforms.Compose([
    RGB2Gray_Transform(),
])

transform_to_lch = transforms.Compose([
    RGB2LCH_Transform(),
])

#####################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ds, dl 생성
train_img_dir = r".\Datasets\RGB2NIR\train"
test_img_dir  = r".\Datasets\RGB2NIR\test"

train_ds = RGB_NIR_Dataset(train_img_dir, transform=transform_basic)
test_ds  = RGB_NIR_Dataset(test_img_dir,  transform=transform_basic)

print('\n'.join([f"{k} {v}" for k, v in dataset_info(train_ds).items()]), "\n", "-"*100)
print('\n'.join([f"{k} {v}" for k, v in dataset_info(test_ds).items()]))
print("="*100)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

print("\n".join([f"{k} {v}" for k, v in dataloader_info(train_dl).items()]))


#####################################################################################################
def train_model(model_gen_, model_dis_, patch_, loss_func_gan_, loss_func_pix_, lambda_pixel_, opt_gen_, opt_dis_,
                color_space_transformer, train_dataloader, exp_no_, num_epochs = 100):
    # 학습
    model_gen_.train()
    model_dis_.train()

    batch_count = 0
    start_time = time.time()

    loss_hist = {'gen':[],
                'dis':[]}

    for epoch in range(num_epochs):
        for rgb, nir, _ in train_dataloader:  # "RGB2NIR"이므로 rgb, nir 순서로 이미지 가져옴. # torch.Size([32, 3, 256, 256])
            batch_size = nir.size(0)
                        
            # real image
            real_rgb_ = rgb.to(device)  # RGB

            if color_space_transformer is None:
                input_img = real_rgb_
            else:
                input_img = color_space_transformer(real_rgb_)

            real_nir = nir.to(device)  # NIR

            # patch label
            real_label = torch.ones(batch_size, *patch_, requires_grad=False).to(device)   # torch.Size([32, 1, 16, 16]) <= [B, 1, P, P]
            fake_label = torch.zeros(batch_size, *patch_, requires_grad=False).to(device)  # torch.Size([32, 1, 16, 16]) <= [B, 1, P, P]
            
            ########## generator ##########
            model_gen_.zero_grad()

            fake_nir = model_gen_(input_img) # 실제 RGB 사진을 입력 받아서 가짜 NIR 이미지 생성, feature extraction을 위한 UNetDown으로 들어가므로 일반적인 GAN처럼 random noize를 주지 않고 바로 real image를 입력으로 줌.
            #fake_nir = std_ * fake_nir + mean_  ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< try result: Bed
        
            # 가짜 이미지에 대한 판별
            out_dis = model_dis_(input_img, fake_nir) # 실제 입력된 이미지(real RGB)를 condition으로 주고, 생성된 가짜 NIR 이미지가 가짜인지 진짜인지를 패치 단위로 예측, # torch.Size([32, 1, 16, 16]) <= [B, 1, P, P]
            
                
            gen_loss = loss_func_gan_(out_dis, real_label)  # nn.BCELoss()
            # pixel-wise loss
            pixel_loss = loss_func_pix_(fake_nir, real_nir) # nn.L1Loss()

            g_loss = gen_loss + (lambda_pixel_ * pixel_loss)
            g_loss.backward()
            opt_gen_.step()

            ########## discriminator ##########
            model_dis_.zero_grad()

            # 판별자에게 진짜 이미지와 가짜 이미지를 5:5로 주어 판별자가 학습할 수 있도록 함.
            # 진짜 이미지에 대한 판별
            out_dis = model_dis_(input_img, real_nir)   # condition과 실제 이미지를 주어서 Discriminator가 각 이미지별 패치 단위로 판별하도록 함함
            real_loss = loss_func_gan_(out_dis, real_label)
            
            # 가짜 이미지에 대한 판별
            out_dis = model_dis_(real_nir, fake_nir.detach())  
            #out_dis = model_dis(input_img, fake_nir.detach()) # <<<<<<<<<<<<<<<< try result : bed  
            fake_loss = loss_func_gan_(out_dis, fake_label)      # nn.BCELoss()

            d_loss = (real_loss + fake_loss) * 0.5
            d_loss.backward()
            opt_dis_.step()

            loss_hist['gen'].append(g_loss.item())
            loss_hist['dis'].append(d_loss.item())

            batch_count += 1
            if batch_count % 100 == 0:
                print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))
    
    color_space_name = "None" if color_space_transformer is None else color_space_transformer.__class__.__name__            
    print(f"\ncolor_space_transformer: {color_space_name}")    
    
    # 모델 (가중치) 저장
    print(f"current experiment_no: {exp_no_}")
    model_dir = f'./RGB2NIR_models/실험No_{exp_no_}/{color_space_name}'
        
    os.makedirs(model_dir, exist_ok=True)
    weights_gen_path = os.path.join(model_dir, 'weights_gen.pt')
    weights_dis_path = os.path.join(model_dir, 'weights_dis.pt')

    torch.save(model_gen_.state_dict(), weights_gen_path)
    torch.save(model_dis_.state_dict(), weights_dis_path)

##################################################################################################################################################
##### test set 전체 이미지에 대한 모델 평가 #####

import torch.nn.functional as F
from Metrics.metrics import calculate_ssim_scores,  calculate_fid_and_psnr, calculate_ndvi 

def evaluate_model(model_gen_, eval_img_dir:str, cs_transformer_, cs_name_, exp_no_):
    print(f"eval_img_dir: {eval_img_dir}")
    eval_ds = RGB_NIR_Dataset(eval_img_dir, transform=transform_basic)   
    eval_dl = DataLoader(eval_ds, batch_size=60, shuffle=False)          # test_set에 있는 이미지의 개수가 60개이므로, batch_size=60을 주어서 모두 가져오게 함.
    
    # evaluation model
    model_gen_.eval()
    print(f"===== color space: {cs_name_} =====\n")
    
    
    # 가짜 이미지 생성해서 진짜 이미지와의 차이 측정
    with torch.no_grad():
        for eval_rgbs, eval_nirs, indices in eval_dl:
            print(f"eval_dl.index: {indices}\n")
            print(f"eval_rgbs.shape: {eval_rgbs.shape}")  # torch.Size([32, 3, 256, 256])
            print(f"eval_nirs.shape: {eval_nirs.shape}")
            
            if cs_transformer_ is None:
                input_imgs = eval_rgbs   # 기본 transform만 적용된 이미지
            else:
                input_imgs = cs_transformer_(eval_rgbs)
                
            fake_nirs = model_gen_(input_imgs.to(device)).detach().cpu()  # 3채널 이미지 반환 

            mse = F.mse_loss(fake_nirs, eval_nirs)
            fid_and_psnr = calculate_fid_and_psnr(fake_nirs, eval_nirs)
            fid_distance = fid_and_psnr["fid"]
            psnr_score   = fid_and_psnr["psnr"]
            avg_ssim     = calculate_ssim_scores(fake_nirs, eval_nirs)
            ndvi_mse, real_ndvis, fake_ndvis = calculate_ndvi(eval_rgbs, eval_nirs, fake_nirs)
            

    ## 측정값 파일에 기록 ##
    now = datetime.now()
    print(now)
    evaluation_result = {'date_time': str(now),
                'color_space': cs_name_,
                'evaluation_mse' : mse.item(),
                'evaluation_PSNR': psnr_score,
                'evaluation_SSIM': avg_ssim.item(),
                'evaluation_FID' : fid_distance.item(),
                'evaluation_ndvi_mse': ndvi_mse.item()}

    eval_log_file_path = rf".\RGB2NIR_models\실험No_{exp_no_}\evaluation_logs.txt"

    with open(eval_log_file_path, "a") as file:
        file.write(json.dumps(evaluation_result))
        file.write("\n")

    print(f"\nreal_ndvis.min(): {real_ndvis.min():.2f}\treal_ndvis.max(): {real_ndvis.max():.2f}")
    print(f"fake_ndvis.min(): {fake_ndvis.min():.2f}\tfake_ndvis.max(): {fake_ndvis.max():.2f}")
    
    return eval_rgbs, real_ndvis, fake_ndvis

########################################################################################################################
def visualize(eval_rgbs_, real_ndvis_, fake_ndvis_, cs_name_, exp_no_, rows=5):
    ## RGB, NIR, fake_NIR 이미지를 한 줄에 시각화하는 fig 생성 및 파일로 저장 ##
    cols = 3
    fig, ax = plt.subplots(rows, cols, figsize=(6,10))

    ndvi_pairs = []
    for i in range(rows):
        r_rgb = eval_rgbs_[i]
        r_ndvi = real_ndvis_[i]
        f_ndvi = fake_ndvis_[i]
        ndvi_pairs.append((r_rgb, r_ndvi, f_ndvi))
        
    for i in range(rows):
        ax[i][0].imshow(ndvi_pairs[i][0].permute(1,2,0), aspect="auto")
        ax[i][0].axis('off')
        
        ax[i][1].imshow(ndvi_pairs[i][1], aspect="auto", cmap='gray')
        ax[i][1].axis('off')

        ax[i][2].imshow(ndvi_pairs[i][2], aspect="auto", cmap='gray')
        ax[i][2].axis('off')
        if i == 0:
            ax[i][0].set_title("Real RGB", fontsize=10)
            ax[i][1].set_title("Real NIR based NDVI", fontsize=10)
            ax[i][2].set_title("Fake NIR based NDVI", fontsize=10)

    fig.suptitle(f"color space: {cs_name_}")
    fig.tight_layout()

    fig_file_path = rf".\RGB2NIR_models\실험No_{exp_no_}\{cs_name_}\result_fig.png"
    plt.savefig(fig_file_path)
    print(f"##### figure saved as a file in {fig_file_path} #####\n")


###################################################################################################################
# Generator, Discrminator 모델 생성
from models.Pix2Pix_RGB2NIR import NIR_GeneratorUNet, Discriminator, initialize_weights

model_gen = NIR_GeneratorUNet(in_channels=3, out_channels=3).to(device)
model_dis = Discriminator(in_channels=3).to(device)

# 손실함수
loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# loss_func_pix 가중치
lambda_pixel = 100

# patch 수
patch = (1,256//2**4,256//2**4)
print(f"patch size: {patch}")



###################################################################################################################
def init_weights():
   # 가중치 초기화 적용
    model_gen.apply(initialize_weights);
    model_dis.apply(initialize_weights);
    
    
    # 최적화 파라미터
    from torch import optim
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    
    opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,beta2))
    opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,beta2))


###################################################################################################################
experiment_no = 1
colos_spaces = [None, RGB2LAB_Transform(), RGB2LCH_Transform(), RGB2YCbCr_Transform("2020"), RGB2YUV_Transform(), RGB2XYZ_Transform()]

for i, cs_transformer in enumerate(colos_spaces):
    init_weights()
    cs_name = "None" if cs_transformer is None else cs_transformer.__class__.__name__
    print(f"\n************ experiment_no: {experiment_no}({i+1}/{len(colos_spaces)}) with color_space: {cs_name} ************")
    train_model(model_gen, model_dis, patch, loss_func_gan, loss_func_pix, lambda_pixel, opt_gen, opt_dis, cs_transformer, train_dl, experiment_no, num_epochs=100)
    evaluate_rgbs, eval_real_ndvis, eval_fake_ndvis = evaluate_model(model_gen, test_img_dir, cs_transformer, cs_name, experiment_no)
    visualize(evaluate_rgbs, eval_real_ndvis, eval_fake_ndvis, cs_name, experiment_no, rows=5)


###################################################################################################################
## 반복 실험 결과 분석 ##

import pandas as pd 

def read_exp_results(exp_no):
    log_path = rf'.\RGB2NIR_models\실험No_{exp_no}\ndvi_result_log.txt'

    if os.path.exists(log_path):
        print(f"log_path: {log_path}")
        
        df = pd.read_json(log_path, lines=True)
        df.columns = ["date_time", "color_space", "MSE", "PSNR", "SSIM", "FID", "NDVI_MSE"]

        df.insert(loc=0, column="exp_no", value = [exp_no]*6)
        return df 
    else:
        print(f"No file exists: {log_path}")

dfs = []
for i in range(1, 5):
    dfs.append(read_exp_results(i+1))
    
df_all = pd.concat(dfs, axis=0)
df_all.reset_index(inplace=True, drop=True)

df_grouped = df_all.groupby('color_space')[['MSE', 'PSNR', 'SSIM', 'FID', 'NDVI_MSE']].mean().reset_index()
