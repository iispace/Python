import torch.nn as nn 

########## ViT Classification Model ##########
class LinearProjection_cls(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super(LinearProjection_cls, self).__init__()
        self.linear_proj   = nn.Linear(patch_vec_size, latent_vec_dim)
        self.cls_token     = nn.Parameter(torch.randn(1, latent_vec_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim))
        self.dropout       = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0) # x: torch.tensor [Batch_size, N, C*p*p]  
        x = x.to(self.linear_proj.weight.dtype) # x와 weight의 dtype 일치시키기 위함.

        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1)
        x += self.pos_embedding 
        x = self.dropout(x)
        return x 
    
class MultiHeadSelfAttention_cls(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate) :
        super(MultiHeadSelfAttention_cls, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads 
        self.latent_vec_dim = latent_vec_dim 
        self.head_dim = int(latent_vec_dim / num_heads) # k 
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.scale = torch.sqrt(self.head_dim *  torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)


    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.T (Transpose of k)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        attention = torch.softmax(q @ k / self.scale, dim=-1)
        x = self.dropout(attention) @ v 
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim) # 각 head에 대한 모든 self-attention 결과가 하나로 concat()되도록 reshape함.

        return x, attention
    
class TFencoder_cls(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super(TFencoder_cls, self).__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiHeadSelfAttention_cls(latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))
    def forward(self, x):
        z = self.ln1(x)
        z, att_map = self.msa(z)
        z = self.dropout(z)
        x = x + z 
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z
        return x, att_map 
    
class VisionTransformer_cls(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes) :
        super(VisionTransformer_cls, self).__init__()
        self.patchembedding = LinearProjection_cls(patch_vec_size=patch_vec_size, num_patches=num_patches,
                                               latent_vec_dim=latent_vec_dim, drop_rate=drop_rate)
        self.transformer    = nn.ModuleList([TFencoder_cls(latent_vec_dim=latent_vec_dim, num_heads=num_heads,
                                                            mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                            for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), nn.Linear(latent_vec_dim, num_classes))

    def forward(self, x):
        """  
            x: torch.tensor [Batch_size, N, C*p*p] torch.float32
        """
        att_list = []
        x = self.patchembedding(x)
        for layer in self.transformer:
            x, att = layer(x)
            att_list.append(att)
        x = self.mlp_head(x[:,0]) # class token만 추출해서 mlp_head에 보냄

        return x, att_list 



########## ViT Regression Model ##########
class LinearProjection_reg(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super(LinearProjection_reg, self).__init__()
        self.linear_proj   = nn.Linear(patch_vec_size, latent_vec_dim) # output: [Batch_size, N, latent_vec_dim]
        self.cls_token     = nn.Parameter(torch.randn(1, latent_vec_dim)) # Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim))
        self.dropout       = nn.Dropout(drop_rate)

    def forward(self, x):
        """ 
            x: one batch, torch.tensor [Batch_size, N, C*p*p]  
        """
        batch_size = x.size(0) 
        x = x.to(self.linear_proj.weight.dtype) # x와 weight의 dtype 일치시키기 위함.

        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1) # [Batch_size, N, D], D=latent_vec_dim
        x += self.pos_embedding 
        x = self.dropout(x)
        return x 
    
class MultiHeadSelfAttention_reg(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate) :
        super(MultiHeadSelfAttention_reg, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads # k
        self.latent_vec_dim = latent_vec_dim # D
        self.head_dim = int(latent_vec_dim / num_heads) # self.head_dim = Dh, num_heads = k, latent_vec_dim = D
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim) # 모든 head의 query를 구하는 것. 즉, multi-head를 한번에 연산하기 위함. D = Dh * k, D=latent_vec_dim
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)   # 모든 head의 key를 구하는 것
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim) # 모든 head의 value를 구하는 것
        self.scale = torch.sqrt(self.head_dim * torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x) # torch.Size[Batch_size, N+1, D]
        k = self.key(x)   # torch.Size[Batch_size, N+1, D]
        v = self.value(x) # torch.Size[Batch_size, N+1, D]

        # 위에서 query, key, value를 모든 head에 대해서 한번에 구했으므로, 아래에서 각 head별 query, key, value로 나눔.
        # latent_vec_dim = num_heads * head_num = k * Dh = D
        # [Batch_size, N+1, D] => [Batch_size, num_heads, N+1, head_dim]으로 변환
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) # head 수를 앞으로 보내기 위해서 벡터의 수(N+1)와 head 수의 순서를 바꿈.
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.T (Transpose of k)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
         
        attention_map = torch.softmax(q @ k / self.scale, dim=-1) # @: 행렬곱 연산자 = torch.matmul()
        x = self.dropout(attention_map) @ v  # x: self-attention
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim) # 각 head에 대한 모든 self-attention 결과가 하나로 concat()되도록 reshape함.

        return x, attention_map
    
class TFencoder_reg(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super(TFencoder_reg, self).__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiHeadSelfAttention_reg(latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))
    def forward(self, x):
        z = self.ln1(x)
        z, att_map = self.msa(z)
        z = self.dropout(z)
        x = x + z 
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z
        return x, att_map 
    
class VisionTransformer_reg(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes=None) :
        super(VisionTransformer_reg, self).__init__()
        self.patchembedding = LinearProjection_reg(patch_vec_size=patch_vec_size, num_patches=num_patches,
                                               latent_vec_dim=latent_vec_dim, drop_rate=drop_rate)
        self.transformer    = nn.ModuleList([TFencoder_reg(latent_vec_dim=latent_vec_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                            for _ in range(num_layers)])
        
        # regression task를 위해 mlp_head 부분을 fc로 수정
        out_feature = 1 # for regression
        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), 
                                      nn.Linear(latent_vec_dim, 64),
                                      nn.LeakyReLU(),
                                      nn.Linear(64, 128),
                                      nn.LeakyReLU(),
                                      nn.Linear(128, 256),
                                      nn.LeakyReLU(),
                                      nn.Linear(256, 128),
                                      nn.LeakyReLU(),
                                      nn.Linear(128, 64),
                                      nn.LeakyReLU(),
                                      nn.Linear(64, 32),
                                      nn.LeakyReLU(),
                                      nn.Linear(32, 16),
                                      nn.LeakyReLU(),
                                      nn.Linear(16, 8),
                                      nn.LeakyReLU(),
                                      nn.Linear(8, out_feature))


    def forward(self, x):
        """  
            x: torch.tensor [Batch_size, N, C*p*p] torch.float32
        """
        att_list = []
        x = self.patchembedding(x)
        for layer in self.transformer:
            x, att = layer(x)
            att_list.append(att)
            
        x = self.mlp_head(x[:,0]) # class token만 추출해서 mlp_head layer에 입력

        return x, att_list 
