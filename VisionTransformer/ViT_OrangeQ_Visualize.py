import numpy as np
import torch
import torch.nn.functional as F 
import torchvision
import matplotlib.pyplot as plt 
import cv2


def minmax(x): 
    m = torch.min(x)
    M = torch.max(x)
    return (M-x)/(M-m)

def visualize_rgb_filters(model, patch_size, num_vis=64, fig_size=(8,8)): 
    """ 
    Args:
        - model: Vision Transformer model
        - patch_size: patch_size, i.e., p
        - num_heads : num_heads, i.e., k
        - num_vis   : 4 ~ D(latent_vec_dim) 개의 범위 중에서 시각화할 latent vector의 개수, 시각화를 위해 제곱수로 할 것!
    """
    linear_embedding = model.patchembedding.linear_proj.weight         # [D, C*p*p]
    print(f"linear_embedding.size(): {linear_embedding.size()} meaning [D, C*p*p]")

    num_channel=int(linear_embedding.size(1) / (patch_size * patch_size))   # 3
    rgb_embedding_filters = linear_embedding.detach().cpu().view(num_channel, patch_size, patch_size, -1).permute(3,0,1,2)
    print(f"rgb_embedding_filters.size(): {rgb_embedding_filters.size()} meaning [D, C, patch_size, patch_size]")
    
    rgb_embedding_filters = minmax(rgb_embedding_filters)

    fig = plt.figure(figsize=fig_size) 

    # D개 중에서 num_vis개만 시각화 
    for i in range(1, num_vis+1):
        rgb = rgb_embedding_filters[i-1].numpy() # [C, p, p]
        ax = fig.add_subplot(int(np.sqrt(num_vis)), int(np.sqrt(num_vis)), i)
        ax.axes.get_xaxis().set_visible(False) # x축 눈금 삭제
        ax.axes.get_yaxis().set_visible(False)
        #ax.imshow(rgb.T)
        ax.imshow(np.transpose(rgb, (1,2,0)))


def visualize_pos_embedding(model, num_patches, fig_size=(8,8)):
    pos_embedding = model.patchembedding.pos_embedding # torch.Size([1, 65, 128]) => [1, N+1, D]
    print(f"pos_embedding.size(): {pos_embedding.size()} meaning [1, N+1, D]")

    fig = plt.figure(figsize=fig_size)
    for i in range(1, pos_embedding.shape[1]):
        sim = F.cosine_similarity(pos_embedding[0, i:i+1], pos_embedding[0, 1:], dim=1) # sim.size() = num_patches
        if i == 1:
            print(f"sim.size(): {sim.size()}")
        reshape_size = (int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))

        sim = sim.reshape(reshape_size).detach().cpu().numpy()
        ax = fig.add_subplot(reshape_size[0],reshape_size[1],i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(sim)
