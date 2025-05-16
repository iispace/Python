import torch 
import kornia
import skimage

class RGB2LCH_Transform:
    def __call__(self, rgb_):
        return self.__rgb2lch(rgb_)
    
    def __rgb2lch(self, rgb_):
        if rgb_.dtype != torch.float32:  # ToTensor()에 의해 torch.Tensor로 변환된 것은 값의 범위가 이미 0 ~ 1로 되어 있음.
            rgb_ = rgb_ / 255.0
        
        rgb_ = rgb_.detach().cpu()
        lab_img = skimage.color.rgb2lab(rgb_.permute(0,2,3,1)) # skimage.color.rbg2lab()함수가 처리할 수 있도록 Channel축을 제일 마지막으로 변경
        lch_img = skimage.color.lab2lch(lab_img)   # numpy.array 로 생성됨.
        lch     = torch.tensor(lch_img).permute(0, 3, 1, 2)            # pytorch로 변환하고 구조를 다시 [B, C, H, W]로 변경
        lch = lch.to("cuda")
        return lch

class RGB2LAB_Transform:
    def __call__(self, rgb_):
        "rgb_ : [B, C, H, W] 형식의 torch Tensor, 값 범위: [0,1]"
        return self.__rgb2lab(rgb_)
    
    def __rgb2lab(self, rgb_):
        # RGB to LAB 변환
        if rgb_.dtype != torch.float32:  # ToTensor()에 의해 torch.Tensor로 변환된 것은 값의 범위가 이미 0 ~ 1로 되어 있음.
            rgb_ = rgb_ / 255.0
        lab = kornia.color.rgb_to_lab(rgb_)
         
        return lab
       
class LAB2RGB_Transform:
    def __call__(self, lab_):
        "lab_ : [B, C, H, W] 형식의 torch Tensor"
        return self.__lab2rgb(lab_)
    
    def __lab2rgb(self, lab_):
        # LAB to RGB 변환
        rgb = kornia.color.lab_to_rgb(lab_)
        
        return rgb 
    
    
class RGB2YUV_Transform:
    def __call__(self, rgb_):
        "rgb_ : [B, C, H, W] 형식의 torch Tensor"
        return self.__rgb2yuv(rgb_)
    
    def __rgb2yuv(self, rgb_):
        # RGB to LAB 변환
        if rgb_.dtype != torch.float32:
            rgb_ = rgb_ / 255.
        yuv = kornia.color.rgb_to_yuv(rgb_)
        return yuv
    
    
class RGB2XYZ_Transform:
    def __call__(self, rgb_):
        "rgb_ : [B, C, H, W] 형식의 torch Tensor"
        
        return self.__rgb2xyz(rgb_)
    
    def __rgb2xyz(self, rgb_):
        # RGB to LAB 변환
        xyz = kornia.color.rgb_to_xyz((rgb_))
        return xyz
    
class RGB2YCbCr_kornia_Transform:
    def __call__(self, rgb_):
        "rgb_ : [B, C, H, W] 형식의 torch Tensor"
        #bgr = kornia.color.rgb_to_bgr(rgb_)
        #bgr = rgb_[:,[2,1,0], :, :]  # Channel 축의 순서를 RGB => BGR로 수정
        return self.__rgb2ycbcr(rgb_)
    
    def __rgb2ycbcr(self, rgb_):
        ycbcr = kornia.color.rgb_to_ycbcr((rgb_))
        return ycbcr  # [B, C, H, W]
    
class RGB2Gray_Transform:
    def __call__(self, rgb_):
        "rgb_ : [B, C, H, W] 형식의 torch Tensor"
        return self.__rgb2gray(rgb_)
    
    def __rgb2gray(self, rgb_):
        if rgb_.dtype != torch.float32:
            rgb_ = rgb_ / 255.
        gray = kornia.color.rgb_to_grayscale(rgb_)
        return gray
    
############################################################################################
class RGB2YCbCr_Transform:
    def __init__(self, BT:str="2020"):
        self.BT = BT # ["601", "709", "2020"] 중 하나
        
    def __call__(self, rgb_:torch.Tensor):
        "[B, C, H, W] 형식의 torch.Tensor 입력"
        return self.__rgb_to_ycbcr(rgb_)
    
    def __rgb_to_ycbcr(self, rgb_:torch.Tensor):
        delta = 0.001
        ycbcr = torch.zeros_like(rgb_)

        if self.BT == "601":
            # BT.601
            ycbcr[:,0,:,:] = 0.299 * rgb_[:,0,:,:] +0.587 * rgb_[:,1,:,:]  + 0.114 * rgb_[:,2,:,:]         ## Y
            ycbcr[:,1] = -0.16874 * rgb_[:,0] - 0.33126 * rgb_[:,1] + 0.5 * rgb_[:,2] + delta              ## Cb 
            ycbcr[:,2] = 0.5 * rgb_[:,0] - 0.41869 * rgb_[:,1] - 0.08131 * rgb_[:,2] + delta               ## Cr

        elif self.BT == "709":
            ycbcr[:,0,:,:] = 0.2126 * rgb_[:,0,:,:] + 0.7152 * rgb_[:,1,:,:]  + 0.0722 * rgb_[:,2,:,:]     ## Y
            ycbcr[:,1] = -0.11457 * rgb_[:,0] - 0.38543 * rgb_[:,1] + 0.5 * rgb_[:,2] + delta              ## Cb 
            ycbcr[:,2] = 0.5 * rgb_[:,0] - 0.45415 * rgb_[:,1] - 0.04585 * rgb_[:,2] + delta               ## Cr
        
        elif self.BT == "2020":
            ycbcr[:,0,:,:] = 0.2627 * rgb_[:,0,:,:] + 0.678 * rgb_[:,1,:,:]  + 0.0593 * rgb_[:,2,:,:]      ## Y
            ycbcr[:,1] = -0.13963 * rgb_[:,0] - 0.36037 * rgb_[:,1] + 0.5 * rgb_[:,2] + delta              ## Cb 
            ycbcr[:,2] = 0.5 * rgb_[:,0] - 0.45979 * rgb_[:,1] - 0.04021 * rgb_[:,2] + delta               ## Cr
       
        return ycbcr 
