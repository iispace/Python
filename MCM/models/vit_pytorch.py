import cv2 

import numpy as np 
import torch 
import torch.nn as nn 
from torch.nn import MSELoss, L1Loss 
from torch.optim import Adam, AdamW, RMSprop
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import ToTensor 
from tqdm import tqdm, trange 

import matplotlib.pyplot as plt 

class LinearProjection(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, dropout):
        """ 
        patch_vec_size: C * patch_size * patch_size
        num_patches   : (H * W) / (patch_size**2)
        latent_vec_dim: latent embedding dimension
        dropout       : dropout rate
        """
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim) #linear_proj.weight.shape: torch.Size([latent_vec, patch_vec_size])
        self.target_token = nn.Parameter(torch.randn(1, latent_vec_dim))
        self.pos_embedding= nn.Parameter(torch.rand(1, num_patches+1, latent_vec_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.to(torch.float32)
        batch_size = x.size(0)
        # 정답 토큰을 linear projection 결과의 첫번째 요소로 추가하기 
        target_tokens = self.target_token.repeat(batch_size, 1,1)  # torch.Size([batch_size, 1, latent_vec])
        x_linear_proj = self.linear_proj(x)                  # torch.Size([batch_size, N, latent_vec])
        x = torch.cat([target_tokens, x_linear_proj], dim=1) # torch.Size([batch_size, N+1, latent_vec])
        # positional embedding 더하기
        x += self.pos_embedding     # torch.Size([batch_size, N+1, latent_vec])
        x = self.dropout(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, dropout):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        
        # multiheads에 대한 query, key, value를 각각 한번에 계산하기 위해 
        # shape을 (latent_vec_dim, head_dim) 대신 (latent_vec_dim, latent_vec_dim)으로 함.
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.key   = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        
        self.rescaler = torch.sqrt(self.head_dim * torch.ones(1)).to(device)  # scalar value 인 head_dim을 크기가 1인 torch.tensor로 만들기 위해 * torch.ones(1)을 사용함.
        self.dropout  = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.to(torch.float32)
        batch_size = x.size(0) 
        q = self.query(x) 
        k = self.key(x)
        v = self.value(x)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)  
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1)  # k.T
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        attention_map = torch.softmax(q @ k /self.rescaler, dim=-1)   # attention_map = weights
        x = self.dropout(attention_map @ v) 
        attention = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim)  # [batch_size, num_patches, latent_vec_dim]
        
        return attention, attention_map
     
class TFencoder(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, dropout, activation_fn):
        """ 
        activation_fn: nn.SELU(), nn.ELU(), nn.GELU(), ...
        """
        super().__init__()
        self.layernorm1 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiHeadAttention(latent_vec_dim, num_heads, dropout)
        self.layernorm2 = nn.LayerNorm(latent_vec_dim)
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                 #nn.GELU(), nn.Dropout(dropout),
                                 activation_fn, nn.Dropout(dropout), 
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        z = self.layernorm1(x)
        z, self_att = self.msa(z)
        z = self.dropout(z)
        x = x + z  # skip connection
        z = self.layernorm2(x)
        z = self.mlp(z)
        x = x + z  # skip connection
        
        return x, self_att
    
    
class ViT(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim,
                 num_heads, mlp_hidden_dim, dropout, num_layers, output_dim, activation_fn):
        """ 
        activation_fn: nn.SELU(), nn.ELU(), nn.GELU(), nn.PReLU()...
        """
        super().__init__()
        if latent_vec_dim % num_heads != 0:
            print(f"latent_vec_dim must be divided by num_heads!!")
            return 
        
        self.embedded_patches    = LinearProjection(patch_vec_size, num_patches, latent_vec_dim, dropout)
        self.transformer_encoder = nn.ModuleList([TFencoder(latent_vec_dim, num_heads, mlp_hidden_dim, dropout, activation_fn)
                                                  for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim),
                                      nn.Linear(latent_vec_dim, output_dim))        
        
    def forward(self, x):
        att_list = [] # Transformer_encoder layer를 하나씩 통과할 때마다 생성된 attention 저장
        x = self.embedded_patches(x)
        for layer in self.transformer_encoder:
            x, att = layer(x)
            att_list.append(att)
        target_tokens = x[:, 0]    
        x = self.mlp_head(target_tokens)
               
        return x, att_list
    
###############################################################################################################################    
from torcheval.metrics.functional import r2_score
import copy

def train_model_ViT(x, y, model, optimizer, loss_fn):

    yhats, att_list = model(x)  # 2D torch.Tensor
    yhats = torch.flatten(yhats) # 1D torch.Tensor
    #yhats = torch.round(yhats)
    
    attentions = torch.stack(att_list).squeeze(1)
    
    model.train()  # enable train mode
    optimizer.zero_grad()
    
    loss = loss_fn(yhats, y)    
    r2 = r2_score(yhats, y)
    
    loss.backward()
    optimizer.step()
    
    return loss, r2, attentions
    

def validate_model_ViT(data_loader, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        loss = 0
        r2 = 0
        list_yhat = []
        list_y = []
        for i, data_dict in enumerate(tqdm(data_loader, disable=True)):
            data       = data_dict['data'].to(device).type(torch.float32)       # torch[Batch_size, num_patches, patch_vec_size]
            targets    = data_dict['target'].to(device).type(torch.float32)     # torch[Batch_size]
            sample_ids = data_dict['sample_id'].type(torch.int32)               # torch[Batch_size]
            
            batch_loss, batch_r2 = 0, 0
            
            yhats, att_list = model(data)
            yhats = torch.flatten(yhats)
            
            batch_loss = loss_fn(yhats, targets)
            batch_r2   = r2_score(yhats, targets)
            
            loss += batch_loss / len(data_loader)
            r2   += batch_r2 / len(data_loader)
            
            list_y.append(targets.cpu())
            list_yhat.append(yhats.cpu()) 
            
        return loss, r2, list_y, list_yhat



def check_model_save_func(args_dict):
    conditions = args_dict['conditions']
    train_loss, test_loss = args_dict['train_loss'], args_dict['test_loss']
    train_r2,   test_r2   = args_dict['train_r2'],   args_dict['test_r2']
    best_acc_train, best_acc_test = args_dict['best_acc_train'], args_dict['best_acc_test']

    list_test_targets, list_test_yhats = args_dict['list_test_targets'], args_dict['list_test_yhats']
    attentions = args_dict['attentions']

    epoch = args_dict['epoch']
    save_acc = args_dict['save_acc']
    model_name = args_dict['model_name']
    data_name = args_dict['data_name']
    arch_id = args_dict['arch_id']
     
    output_dict = {}
    output_dict['save_acc'] = args_dict['save_acc']
    
    IsConditionMet = False
    if 1 in conditions:
        if args_dict['curr_acc_diff'] <= 0.2 and test_r2 > best_acc_test and train_r2 > best_acc_train:
            IsConditionMet = True
    if 2 in conditions:
        if args_dict['train_r2'] >= 0.5 and test_r2 >=0.49:
            IsConditionMet = True
       
    
    if IsConditionMet:
        model = args_dict['model']
        output_dict['best_model_weights'] = copy.deepcopy(model.state_dict())
        if test_r2 > save_acc:
            torch.save(model.state_dict(), rf".\trained_models\{data_name}\{arch_id}\{model_name}_Best_Model_{train_r2:.2f}_{test_r2:.2f}.pth")
            output_dict['save_acc'] = test_r2 
            print(f"<==== model saved!!")
            
        output_dict['best_attentions'] = attentions
        output_dict['best_acc_test'] = test_r2 
        output_dict['best_acc_train'] = train_r2 
        output_dict['model_updated'] = True 
        output_dict['best_ys'] = list_test_targets 
        output_dict['best_yhats'] = list_test_yhats 
        
        output_best_perform_dict = {}
        output_best_perform_dict['epoch'] = epoch 
        output_best_perform_dict['train_loss'], output_best_perform_dict['train_acc'] = train_loss.detach().cpu().item(), train_r2.cpu().item()
        output_best_perform_dict['test_loss'],  output_best_perform_dict['test_acc']  = test_loss.detach().cpu().item(), test_r2.cpu().item()
        print(f"<====== best model updated!!")
        output_dict['best_perform_dict'] = output_best_perform_dict
    
    return output_dict


def eval_final_model(train_dl, test_dl, model, loss_fn, device, best_model_weights=None):
    print()
    title = " --- final model ---"
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        title = " ..... best_model_weights reloaded ....."
        
    print(f"{title}")
    final_train_loss, final_train_r2, final_list_train_targets, final_list_train_yhats = validate_model_ViT(train_dl, model, loss_fn, device)
    final_test_loss, final_test_r2, final_list_test_targets, final_list_test_yhats = validate_model_ViT(test_dl, model, loss_fn, device)
    
    final_acc_diff = np.abs(final_train_r2.detach().cpu() - final_test_r2.detach().cpu())
    
    eval_dict = {}
    eval_dict['train_loss'] = final_train_loss
    eval_dict['train_r2'] = final_train_r2
    eval_dict['test_loss'] = final_test_loss
    eval_dict['test_r2'] = final_test_r2
    eval_dict['acc_diff'] = final_acc_diff
    
    perf_msg = f"[Best/Final Model] train_loss: {final_train_loss:.5e}, train_r2: {final_train_r2:.4f}"
    perf_msg += f"\n[Best/Final Model] test_loss: {final_test_loss:.5e}, test_r2: {final_test_r2:.4f}"
    msg = f"test_targets: {final_list_test_targets}\n"
    msg += f"test_yhats: {final_list_test_yhats}\n"
    print(msg)
    print(perf_msg)
    
    return eval_dict
###############################################################################################################################    
import torch 
import torchvision 
from torch.utils.data import Dataset 
from typing import List, Dict, Tuple, Any

import torchvision.transforms.functional 

class Research_Dataset_FlattenPatches(Dataset):
    def __init__(self, mcm, sample_idxs, targets, mtx_size, patch_size):
        """ 
        Args:
            mcm        : 3D torch.tensor(3 channel matrixified data) of which shape is (N, C, H, W)
            sample_idxs: list of sample indices (not sample sequence)
            targets    : pd.DataFrame, shape: (N, 1)
            mtx_size   : H or W
            patch_size : patch_size (default=8)
            train      : True if the given data is for 'train' set else False
        """
        self.mcm = mcm   # torch.Size([N, C, H, W])
        self.sample_idxs = sample_idxs
        self.mtx_size = mtx_size if mtx_size % 2 == 0 else mtx_size+1  # add 1 if mtx_size is not even
        self.patch_size = patch_size 
        self.targets = targets
        
        
    def __patchification__(self, matrices:torch.tensor):
        """ 
            matrices: 3D torch tensor (C, H, W)
        """
        num_ch = matrices.size(0)
        patches = matrices.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size).reshape(num_ch, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1,0,2,3) # [num_patches, C, patch_size, patch_size]
        num_patches = patches.size(0)      # num_patches = (mtx_size * mtx_size) / patch_size**2
        return (num_patches, patches.reshape(num_patches, -1))  # (num_patches, C*patch_size*patch_size)

    def __len__(self):
        return len(self.mcm)    # return number of samples
    
    def __getitem__(self, index: int) -> Tuple[List[List[List[torch.FloatTensor]]], float, int]:
        """
        Args:
            index (int): sample sequence (not sample id)

        Returns:
            Tuple[List[List[List[torch.FloatTensor]]], float, int]: (3D array of torch.FloatTensor, target value(float), sample_id(int))
        """
        resized_data = torchvision.transforms.functional.resize(self.mcm[index], self.mtx_size, antialias=False)
        (self.num_patches, data) = self.__patchification__(resized_data)
        target = self.targets.iloc[index, 0]
        sample_id = self.sample_idxs[index]
        sample = {'data': data, 'target': target, 'sample_id': sample_id}
        
        return sample    
