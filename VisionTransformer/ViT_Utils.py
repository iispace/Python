class PatchGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size 

    def __call__(self, img):
        num_channels = img.size(0)
        """
        Arg:
            - img : 3D Tensor [C, H, W]
        """
        # img.unfold(dimension, size_of_each_slice, stride): img를 dimension 방향으로 자르는 함수 (dimension=1: dimension[1] in the img, dimension=2: dimension[2] in the img)
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size).reshape(num_channels, -1, self.patch_size, self.patch_size)  # [C, N, p, p], C: Num_channel, N: Num_patches, p: patch_size
        patches = patches.permute(1,0,2,3)  # [C, N, p, p] to [N, C, p, p]
        num_patch = patches.size(0)

        return patches.reshape(num_patch, -1)  # [N, C*p*p]
