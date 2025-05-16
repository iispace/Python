from os import listdir 
from os.path import join 
from PIL import Image 
from torch.utils.data import Dataset 



class RGB_NIR_Dataset(Dataset):
    def __init__(self, img_root_dir, transform=False):
        """  
        -img_root_dir\\RGB 폴더에 있는 이미지 파일의 이름과
        -img_root_dir\\NIR 폴더에 있는 이미지 파일의 이름이 서로 동일해야 하며 서로 짝을 이루어야 함.        
                
        -transform : 여기에 인자로 들어오는 transform은 ToTensor(), Normailize(), Resize() 와 같이 RGB와 NIR에 공통 적용되는 변환만 사용하기
        """
        super().__init__()
        self.dir_RGB = join(img_root_dir, 'RGB')    
        self.dir_NIR = join(img_root_dir, 'NIR')
        self.img_filenames = [fn for fn in listdir(self.dir_RGB)]
        self.transform = transform

    def __getitem__(self, index):
        RGB = Image.open(join(self.dir_RGB, self.img_filenames[index])).convert('RGB')
        NIR = Image.open(join(self.dir_NIR, self.img_filenames[index])).convert('RGB')  # 3 채널로 만들기 위함.
        
        if self.transform:
            RGB = self.transform(RGB)
            NIR = self.transform(NIR)

        return RGB,NIR, index

    def __len__(self):
        return len(self.img_filenames)
    
    
#########################################################################################

def dataset_info(ds_):
    keys = list(ds_.__dict__.keys())
    len_ = len(ds_.img_filenames)
    dir_RGB = ds_.dir_RGB
    dir_NIR = ds_.dir_NIR
    transform_ = ds_.transform
    return {"keys": keys, "len": len_, "dir_RGB": dir_RGB, "dir_NIR": dir_NIR, "transform": transform_}

def dataloader_info(dl_):
    len_ = len(dl_)
    batch_size = dl_.batch_size
    num_workers = dl_.num_workers
    num_imgs = len(dl_.dataset)
    img_shape = dl_.dataset[num_imgs-1][0].shape
    
    dataset_keys = list(dl_.dataset.__dict__.keys())
    
    return {"len(dl)": len_, "dl.batch_size": batch_size, "dl.num_workers": num_workers, "len(dl.dataset)": num_imgs, "dl.dataset[len(dl.dataset)][0]": img_shape, "dataset_keys": dataset_keys}
   
