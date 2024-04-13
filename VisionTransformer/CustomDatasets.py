import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils, io
from torchvision.transforms import transforms 

from typing import Dict, List, Tuple 

import warnings
warnings.filterwarnings('ignore')

class OrangeQualityDataset(Dataset):
    def __init__(self, root_dir: str, img_size:int = 12):
        """ 
        Args:
            - root_dir : Root Dir for image files              
        """
        self.root_dir = root_dir
        self.img_size = img_size

        self.df = self._read_img_file_path()
        self.sample_idx = self.df.index.tolist()

        self.data = self._read_img()
        self.labels  = self.df['label']
        self.classes = self.df['label'].unique().tolist()   
        self.targets = self._get_targets()
        self.transform = transform

    def _read_img_file_path(self) -> pd.DataFrame:
        df = pd.DataFrame(["image_path", "label"])
        walks = list(os.walk(self.root_dir))  
        top_level = walks[0]
        root = top_level[0]
        label_dirs = top_level[1]
        
        sample_file_list = []
        sample_label_list = []
        sample_id_list = []
      
        for i in range(1, len(walks)):
            sub_level = walks[i] # (root, dir, file) 구조의 tuple
            files = sub_level[2] 
            sample_id = [int(name.split(".")[0]) for name in files]
            label = (label_dirs[i-1])
            sample_file_list.extend(files)
            sample_label_list.extend([label] * len(files))
            sample_id_list.extend(sample_id)
        df = pd.DataFrame(data={"image_path": sample_file_list, "label": sample_label_list}, index=sample_id_list)
        df = df.sort_index() # sample_id 순으로 오름차순 정렬
        return df

    def _get_targets(self) -> List[int]:
        targets = []
        label_onehot_df = pd.get_dummies(self.df, columns=['label'])
        columns = label_onehot_df.columns.tolist()
        for _, row in label_onehot_df.iloc[:, 1:].iterrows(): # row: pd.Series (Series.index = columns name)
            output = row[row].index
            column_idx = columns.index(output[0])
            targets.append(column_idx) 
        return targets

    def _read_img(self):
        labels = self.df['label'] # pd.Series (index: sample_id, value: label)
        file_names = self.df['image_path']
        file_paths = [f"{self.root_dir}\\{label}\\{file_name}" for label, file_name in zip(labels, file_names)]
        #print(f"file to read: {file_paths}")
        data = np.full((self.df.shape[0], self.img_size, self.img_size, 3), np.nan, dtype=int)
        for i, file_path in enumerate(file_paths):
            img = io.read_image(file_path)                # torchvision.transforms.io.read_image() returns 3D Tensor[C, H, W]
            img_permute = torch.permute(img, (1,2,0))     # convert structure from [C, H, W] to [H, W, C]
            img_BGR = img_permute.numpy().astype('uint8') # Tensor to Numpy array with dtype('uint8')
            #img_RGB = img_BGR[..., [2,1,0]]              # Color convert from BGR to RGB (for torch.tensor object ) 
            img_RGB = img_BGR[...,::-1]                   # Color convert from BGR to RGB (for numpy.ndarray object )
            data[i] = img_RGB    
        return data

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        """ 
        Args:
            - idx: 1D torch.tensor([], dtype=torch.int) or int
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        
        image = self.data[idx]
        sample_id = self.sample_idx[idx]
        label = self.labels[self.labels.index==sample_id].values[0]
        target = self.targets[idx]
        sample = (image, target, label, sample_id)

from typing import Any, Callable, Optional, Tuple, Union
import pickle 
from PIL import Image 

#########################################################################################################################

from typing import Any, Callable, Optional, Tuple, Union
import pickle 
from PIL import Image 

class OrangeQualityDataset2(Dataset):
    def __init__(self, root_dir: str, train=True, transform=None, target_transform=None):
        """ 
        Args:
            - root_dir : 이미지 Root 경로
            - transform(호출가능한 함수, 선택적 매개변수) : 샘플에 적용될 수 있는 선택적 변환
        """
        self.base_folder = root_dir 
        self.train=train 
        self.image_path = os.path.join(self.base_folder, "train") if self.train else os.path.join(self.base_folder, "test")
        self.classes = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.file_paths = self._get_file_path()
        self.data:Any = []
        self.labels = []
        self.targets = []
        self.sample_ids = []

        for i, file_path in enumerate(self.file_paths):
            label = float(os.path.dirname(file_path).split("\\")[-1])
            basename = int(os.path.basename(file_path).split('.')[0]) # file name is the sample_id
            img = io.read_image(file_path)            # torchvision.transforms.io.read_image() returns 3D Tensor[C, H, W]
            img_permute = torch.permute(img, (1,2,0)) # convert structure from [C, H, W] to [H, W, C]
            img_BGR = img_permute.numpy().astype('uint8') # Tensor to Numpy array with dtype('uint8')
            #img_RGB = img_BGR[..., [2,1,0]]          # for torch.tensor object
            img_RGB = img_BGR[...,::-1]               # Color convert from BGR to RGB (numpy.ndarray: [H,W,C])
            self.data.append(img_RGB)                 # self.data: list of numpy.ndarray
            self.labels.append(label)
            target = self.classes.index(label)
            self.targets.append(target)
            self.sample_ids.append(basename)

        self.img_size = self.data[0].shape[0]

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        """ 
        Args:
            - index: Index
        Returns:
            - tuple: (image, target) where target is index of the target class
        """
        img = self.data[index]  # numpy.ndarray
        target = self.classes.index(self.labels[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img) 

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
         

    def print_file_paths(self):
        for i, file_path in enumerate(self.file_paths):
            print(f"[{i:>3}]: {file_path}")

    def _get_file_path(self):
        file_path_df = self._read_img_file_path()
        labels = file_path_df['label'] # pd.Series (index: sample_id, value: label)
        file_names = file_path_df['image_path']
        file_paths = [f"{self.image_path}\\{label}\\{file_name}" for label, file_name in zip(labels, file_names)]
        return file_paths

    def _read_img_file_path(self) -> pd.DataFrame:
        df = pd.DataFrame(["image_path", "label"])
        walks = list(os.walk(self.image_path)) # 폴더 개수 크기의 walk 리스트 반환. 각 walk는 (root, dir, file) 구조의 tuple
        top_level = walks[0]
        root = top_level[0]
        label_dirs = top_level[1]
        # root_dir 바로 아래(top_level)에 있는 file 목록(top_level[2])은 사용하지 않으므로 생략
        sample_file_list = []
        sample_label_list = []
        sample_id_list = []
        for i in range(1, len(walks)):
            sub_level = walks[i] # (root, dir, file) 구조의 tuple
            files = sub_level[2] 
            sample_id = [int(name.split(".")[0]) for name in files]
            label = (label_dirs[i-1])
            sample_file_list.extend(files)
            sample_label_list.extend([label] * len(files))
            sample_id_list.extend(sample_id)
        df = pd.DataFrame(data={"image_path": sample_file_list, "label": sample_label_list}, index=sample_id_list)
        df = df.sort_index() # sample_id 순으로 오름차순 정렬
        return df    


