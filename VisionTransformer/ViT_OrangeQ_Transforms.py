from scipy.ndimage import zoom  # image resize에 사용할 library
import torch  

class OrangeQ_Resize:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        """
        Arg:
            - img : 3D tensor [H, W, C] <class 'torch.Tensor'> torch.Size([12, 12, 3])
        """
        cur_img_size = img.size()[0] 
        resized_img = zoom(img, (self.img_size/cur_img_size, self.img_size/cur_img_size, 1)) # [H,W,C] => [H,W,C]구조로 resize. 결과: numpy.ndarray
        resized_img = torch.tensor(resized_img)
        
        return resized_img  # returns: [H, W, C] torch.tensor
    
class OrangeQ_PatchGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size 

    def __call__(self, img):
        num_channels = img.size(2) # <= 넘겨지는 img tensor(OrangeQuality image) 구조가 [H, W, C]이므로, img.size(2)를 가져옴 
        """
        Arg:
            - img : 3D torch.tensor [H, W, C]

        img.unfold(dimension, size_of_each_slice, stride): img를 dimension 방향으로 자르는 함수 
                                                           ([H,W,C] 구조인 경우 dimension=0: H 방향, dimension=1: W 방향)
        """

        # patches: # [C, N, p, p], C: Num_channel, N: Num_patch, p: patch_size
        patches = img.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size).reshape(num_channels, -1, self.patch_size, self.patch_size)

        patches = patches.permute(1,0,2,3)    # [C, N, p, p] to [N, C, p, p]  ex: [3,256,16,16] to [256,3,16,16]
        num_patch = patches.size(0)

        return patches.reshape(num_patch, -1) # [N, C*p*p] ex: [256, 768] torch.tensor
