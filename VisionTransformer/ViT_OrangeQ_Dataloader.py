from ViT_OrangeQ_Transforms import OrangeQ_Resize, OrangeQ_PatchGenerator
from ViT_Datasets import OrangeQualityDataset

import torch 
import numpy as np 
import torchvision
from torchvision.transforms import transforms 
from torch.utils.data import WeightedRandomSampler, DataLoader

class ImgPatcherNDataLoader:
    def __init__(self, IMAGE_ROOT=None, dataname='OrangeQuality', patch_size=16, img_size=256, batch_size=64):
        self.patch_size = patch_size 
        self.dataname = dataname 
        self.img_size = img_size
        self.batch_size = batch_size 
        self.IMAGE_ROOT = IMAGE_ROOT # 이미지 파일이 있는 최상위 경로 (예: D:\OrangeQuality)

    def make_weights(self, labels, nclasses):
        labels = np.array(labels)  # labels: numpay.ndarray([192])
        weight_list = []
        for cls in range(nclasses): # cls = [0,1,2,3,4,5,6,7]
            idx = np.where(labels == cls)[0]
            count = len(idx)
            weight = 1 / count if count > 0 else 0 # count=0일때 오류 방지 트릭
            weights = [weight] * count # weight를 값으로 갖는 요소를 count 만큼 늘림. 예: weight=0.25이고 count=4라면, weights=[0.25,0.25,0.25,0.25] 
            weight_list += weights
        #print(f"[make_weights] {len(weight_list)} weight_list: {weight_list}")
        return weight_list 

    
    def patchdata(self):
        train_transform = transforms.Compose([OrangeQ_Resize(self.img_size), OrangeQ_PatchGenerator(self.patch_size)])
        test_transform  = transforms.Compose([OrangeQ_Resize(self.img_size), OrangeQ_PatchGenerator(self.patch_size)])
        
        if self.dataname == 'OrangeQuality':
            trainset = OrangeQualityDataset(self.IMAGE_ROOT, train=True, transform=train_transform)
            testset  = OrangeQualityDataset(self.IMAGE_ROOT, train=False, transform=test_transform)

        elif self.dataname == 'imgagenet':
            pass 

        # trainset.classes: 클래스 종류 list (예: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
        # trainset.targets: 정답 list (예: [1,9,2,3,4,4...])
        print(f"len(trainset.targets): {len(trainset.targets)}, type(trainset.targets): {type(trainset.targets)}trainset.targets: {trainset.targets}")
        print(f"len(trainset.classes: {len(trainset.classes)}, type(trainset.classes): {type(trainset.classes)} {trainset.classes}")
        
        weights = self.make_weights(trainset.targets, len(trainset.classes)) # 가중치 계산
        weights = torch.DoubleTensor(weights)
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) # class가 imbalance해도 똑같은 확률로 sampling하는 방법
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True) # class가 imbalance해도 똑같은 확률로 sampling하는 방법
        trainloader = DataLoader(trainset, batch_size=self.batch_size, sampler=sampler, shuffle=False, num_workers=0) #
        testloader  = DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return trainloader, testloader
