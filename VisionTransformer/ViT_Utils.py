import matplotlib.pyplot as plt 
import numpy as np 
import torch 
import torchvision
from  torchvision import transforms
from torch.utils.data import sampler, DataLoader

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


class Flattened2Dpatches:
    def __init__(self, patch_size=16, dataname='imagenet', img_size=256, batch_size=64):
        self.patch_size = patch_size 
        self.dataname = dataname 
        self.img_size = img_size
        self.batch_size = batch_size 

    def make_weights(self, labels, nclasses):
        labels = np.array(labels)
        weight_list = []
        for cls in range(nclasses):
            idx = np.where(labels == cls)[0]
            count = len(idx)
            weight = 1 / count if count > 0 else 0 # count=0일때 오류 방지 트릭
            weights = [weight] * count
            weight_list += weights
        return weight_list 
    
    def patchdata(self):
        mean = (0.4914, 0.4822, 0.4465) # cifar10 데이터셋 기준 채널별 평균값
        std = (0.2023, 0.1194, 0.2010)  # cifar10 데이터셋 기준 채널별 표준편차값
        train_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.RandomCrop(self.img_size, padding=2),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std),
                                              PatchGenerator(self.patch_size)])
        test_transform = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor(),
                                             transforms.Normalize(mean, std), PatchGenerator(self.patch_size)])
        
        if self.dataname == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./Img_Data', train=True,  download=True, transform=train_transform)
            print(f"dataname: {self.dataname}, trainset.data: {type(trainset.data)} {trainset.data.shape}") # Numpy.ndarray [Num_sample, H, W, C]
            testset  = torchvision.datasets.CIFAR10(root='./Img_Data', train=False, download=True, transform=test_transform)
            evens = list(range(0, len(testset), 2))  # 짝수 인덱스
            odds = list(range(1, len(testset), 2))   # 홀수 인덱스
            valset  = torch.utils.data.Subset(testset, evens)
            testset = torch.utils.data.Subset(testset, odds)
        elif self.dataname == 'OrangeQuality':
            IMAGE_ROOT = rf"D:\Image\OrangeQuality" # local folder에서 가져오기
            train_root = os.path.join(IMAGE_ROOT, "train")
            test_root  = os.path.join(IMAGE_ROOT, "test")
            print(f"train_root: {train_root}")
            print(f"test_root : {test_root}")

            trainset = OrangeQualityDataset(train_root, img_size=12)
            #testset = OrangeQualityDataset(test_root, img_size=12)
            print(f"dataname: {self.dataname}, trainset.data: {type(trainset.data)} {trainset.data.shape}") # Numpy.ndarray [Num_sample, H, W, C]

            evens = list(range(0, 49, 2)) # 49: number of samples in testset
            odds  = list(range(1, 49, 2)) # 49: number of samples in testset
            valset  = OrangeQualityDataset(test_root, img_size=12, seq_indices=evens)
            testset = OrangeQualityDataset(test_root, img_size=12, seq_indices=odds)
        elif self.dataname == 'imgagenet':
            pass 

        # trainset.classes: 클래스 종류 list (예: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
        # trainset.targets: 정답 list (예: [1,9,2,3,4,4...])
        print(f"len(trainset.targets): {len(trainset.targets)}, type(trainset.targets): {type(trainset.targets)}trainset.targets: {trainset.targets}")
        print(f"len(trainset.classes: {len(trainset.classes)}, type(trainset.classes): {type(trainset.classes)} {trainset.classes}")
        
        weights = self.make_weights(trainset.targets, len(trainset.classes)) # 가중치 계산
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) # class가 imbalanced dataset에서 같은 확률로 sampling하기 위해 생성한 sampler
        trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=0) #
        valloader  = DataLoader(valset,  batch_size=self.batch_size, shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return trainloader, valloader, testloader

def inv_normal(img):
    mean = (0.4914, 0.4822, 0.4465) # cifar10 데이터셋 기준 채널별 평균값
    std = (0.2023, 0.1194, 0.2010)  # cifar10 데이터셋 기준 채널별 표준편차값

    img_inv = torch.tensor(np.full(img.size(), np.nan))
    print(f"img_inv.size(): {img_inv.size()}")
    for i in range(3):
        #img[:, i, :, :] = torch.abs(img[:, i, :, :]*std[i] + mean[i])
        img_inv[:, i, :, :] = torch.abs(img[:, i, :, :]*std[i] + mean[i])
    return img_inv

def imshow(img):
    plt.figure(figsize=(8,8))   
    plt.imshow(img.permute(1,2,0).numpy())
    plt.axis('off')
    #plt.show()
    #plt.close()
    return img
