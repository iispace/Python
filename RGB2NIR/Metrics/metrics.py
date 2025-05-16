import torch
import torch.nn.functional as F
from torcheval.metrics import PeakSignalNoiseRatio, FrechetInceptionDistance

from torchmetrics.functional import structural_similarity_index_measure as ssim
# Assume each image is [C, H, W] and values are in [0, 1]
# fake_imgs and real_imgs: torch.Tensor of shape [B, C, H, W]
# Example: fake_imgs = torch.rand(16, 3, 256, 256), real_imgs = torch.rand(16, 3, 256, 256)

def calculate_ssim_scores(fake_imgs, real_imgs):
    # Ensure images are in range [0, 1]
    fake_imgs = fake_imgs.clamp(0, 1)
    real_imgs = real_imgs.clamp(0, 1)

    # Calculate SSIM per image pair and take average
    scores = [ssim(f[None, ...], r[None, ...]) for f, r in zip(fake_imgs, real_imgs)]
    return torch.stack(scores).mean()   # 값이 클수록(1에 가까울수록) 우수


def calculate_fid_and_psnr(fake_imgs, real_imgs):
  fid = FrechetInceptionDistance()
  fid.update(real_imgs.clamp(0,1), is_real=True)
  fid.update(fake_imgs.clamp(0,1), is_real=False)
  fid_distance = fid.compute()
  print(f"FID: {fid_distance:.2f}")  # 값이 작을수록 우수

  psnr = PeakSignalNoiseRatio()
  psnr.update(real_imgs, fake_imgs)
  psnr_score = psnr.compute().item()
  print(f"PSNR: {psnr_score:.2f}\n")  # 값이 클수록 우수
  
  return {"fid" fid_distance, "psnr": psnr}


def calculate_ndvi(rgb_imgs, real_nirs, fake_nirs):
    real_nir_channels = real_nirs[:, 0]
    fake_nir_channels = fake_nirs[:, 0]
    rgb_red_channels = rgb_imgs[:, 0]
    
    eps = 1e-6
    real_ndvi = (real_nir_channels - rgb_red_channels) / (real_nir_channels + rgb_red_channels + eps) 
    fake_ndvi = (fake_nir_channels - rgb_red_channels) / (fake_nir_channels + rgb_red_channels + eps) 
    
    print(f"real_ndvi.shape: {real_ndvi.shape}")
    print(f"fake_ndvi.shape: {fake_ndvi.shape}")
    
    ndvi_mse = F.mse_loss(real_ndvi, fake_ndvi)  # 값이 작을수록 우수
    print(f"\nndvi_mse : {ndvi_mse.item():.2f}")

    return ndvi_mse
