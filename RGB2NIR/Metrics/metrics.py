import torch
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
