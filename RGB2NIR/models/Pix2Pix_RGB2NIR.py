import torch 
import torch.nn as nn 

# UNet
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x


##############################################################################################
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)
        #print(f"self.up: {self.up}")

    def forward(self, x, skip):      # 입력으로 주어진 x와 skip의 shape이 각각 x.shape = [16, 128, 64, 64], skip.shape = [16, 64, 128, 128] 이라고 할 때, 
        x = self.up(x)               # torch.Size([16, 64, 128, 128]) <= nn.ConvTranspose2d()의 out_channels=64이고, input=64, stride=2, padding=1, kernel_size=4이므로, [16,64,128,128] 출력
        x = torch.cat((x,skip), dim=1)    # x tensor와 skip tensor를 C 단에서 합침  <= [P, C, H, W] 구조의 shape에서 dim=1은 C 부분을 의미미
        return x

###############################################################################################
# 가짜 NIR 이미지 생성 모델    
class NIR_GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, out_channels=64, normalize=False)  # UNetDown은 Conv2d() 연산: feature map 생성
        #print(f"self.down1: {self.down1}")
        self.down2 = UNetDown(64,128)                 
        self.down3 = UNetDown(128,256)               
        self.down4 = UNetDown(256,512,dropout=0.5) 
        self.down5 = UNetDown(512,512,dropout=0.5)      
        self.down6 = UNetDown(512,512,dropout=0.5)             
        self.down7 = UNetDown(512,512,dropout=0.5)              
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5)

        self.up1 = UNetUp(512,512,dropout=0.5)                                # UNetUp은 ConvTranspose2d() 연산: Upsampling 생성
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512,dropout=0.5)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                                 nn.Tanh()
                                )

    def forward(self, x):
        d1 = self.down1(x)   # torch.Size([16, 64, 128, 128]) <= self.down1()의 out_channels=64이고, input=256, kernel=4, stride=2, padding=1 이므로, [16, 64, 128, 128] 출력
        d2 = self.down2(d1)  # torch.Size([16, 128, 64, 64]) <= self.down1()의 out_channels=128이고, input=128, kernel=4, stride=2, padding=1 이므로, [16, 128, 64, 64] 출력
        d3 = self.down3(d2)  # torch.Size([16, 256, 32, 32]) <= self.down1()의 out_channels=256이고, input=64, kernel=4, stride=2, padding=1 이므로, [16, 256, 32, 32] 출력
        d4 = self.down4(d3)  # torch.Size([16, 512, 16, 16]) <= self.down1()의 out_channels=512이고, input=32, kernel=4, stride=2, padding=1 이므로, [16, 512, 16, 16] 출력
        d5 = self.down5(d4)  # torch.Size([16, 512, 8, 8])   <= self.down1()의 out_channels=512이고, input=16, kernel=4, stride=2, padding=1 이므로, [16, 512, 8, 8] 출력
        d6 = self.down6(d5)  # torch.Size([16, 512, 4, 4])   <= self.down1()의 out_channels=512이고, input=8, kernel=4, stride=2, padding=1 이므로, [16, 512, 4, 4] 출력
        d7 = self.down7(d6)  # torch.Size([16, 512, 2, 2])   <= self.down1()의 out_channels=512이고, input=4, kernel=4, stride=2, padding=1 이므로, [16, 512, 2, 2] 출력
        d8 = self.down8(d7)  # torch.Size([16, 512, 1, 1])   <= self.down1()의 out_channels=512이고, input=2, kernel=4, stride=2, padding=1 이므로, [16, 512, 1, 1] 출력

        u1 = self.up1(d8,d7) # d8 tensor와 d7 tensor가 C(channel) 단에서 합쳐짐(skip connection 구현).
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)     # u8: torch.Size([16, 3, 256, 256])

        out = self.__generate_nir(u8)
        return out
    
    def __generate_nir(self, u_out):
        """_summary_

        Args:
            u_out (torch.Tensor): output of UNet [B, C, H, W]
        """
        # 이미지별로 3채널의 평균 구하기
        mean_hw_across_CH = torch.mean(u_out, dim=1)  # [B, H, W]
        # convert [B, H, W] to [B, 1, H, W]
        mean_C_ = mean_hw_across_CH.unsqueeze(dim=1)
        # 이미지별로 [H,W]를 3번 복사해서 3채널로 만들기
        generated_nirs = mean_C_.expand(-1, 3, -1, -1)
        
        return generated_nirs

###########################################################################################################################
# 가짜 이미지 생성 모델
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, out_channels=64, normalize=False)  # UNetDown은 Conv2d() 연산: feature map 생성
        #print(f"self.down1: {self.down1}")
        self.down2 = UNetDown(64,128)                 
        self.down3 = UNetDown(128,256)               
        self.down4 = UNetDown(256,512,dropout=0.5) 
        self.down5 = UNetDown(512,512,dropout=0.5)      
        self.down6 = UNetDown(512,512,dropout=0.5)             
        self.down7 = UNetDown(512,512,dropout=0.5)              
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5)

        self.up1 = UNetUp(512,512,dropout=0.5)                                # UNetUp은 ConvTranspose2d() 연산: Upsampling 생성
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512,dropout=0.5)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                                 nn.Tanh()
                                )

    def forward(self, x):
        d1 = self.down1(x)   # torch.Size([16, 64, 128, 128]) <= self.down1()의 out_channels=64이고, input=256, kernel=4, stride=2, padding=1 이므로, [16, 64, 128, 128] 출력
        d2 = self.down2(d1)  # torch.Size([16, 128, 64, 64]) <= self.down1()의 out_channels=128이고, input=128, kernel=4, stride=2, padding=1 이므로, [16, 128, 64, 64] 출력
        d3 = self.down3(d2)  # torch.Size([16, 256, 32, 32]) <= self.down1()의 out_channels=256이고, input=64, kernel=4, stride=2, padding=1 이므로, [16, 256, 32, 32] 출력
        d4 = self.down4(d3)  # torch.Size([16, 512, 16, 16]) <= self.down1()의 out_channels=512이고, input=32, kernel=4, stride=2, padding=1 이므로, [16, 512, 16, 16] 출력
        d5 = self.down5(d4)  # torch.Size([16, 512, 8, 8])   <= self.down1()의 out_channels=512이고, input=16, kernel=4, stride=2, padding=1 이므로, [16, 512, 8, 8] 출력
        d6 = self.down6(d5)  # torch.Size([16, 512, 4, 4])   <= self.down1()의 out_channels=512이고, input=8, kernel=4, stride=2, padding=1 이므로, [16, 512, 4, 4] 출력
        d7 = self.down7(d6)  # torch.Size([16, 512, 2, 2])   <= self.down1()의 out_channels=512이고, input=4, kernel=4, stride=2, padding=1 이므로, [16, 512, 2, 2] 출력
        d8 = self.down8(d7)  # torch.Size([16, 512, 1, 1])   <= self.down1()의 out_channels=512이고, input=2, kernel=4, stride=2, padding=1 이므로, [16, 512, 1, 1] 출력

        u1 = self.up1(d8,d7) # d8 tensor와 d7 tensor가 C(channel) 단에서 합쳐짐(skip connection 구현).
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)     # u8: torch.Size([16, 3, 256, 256])

        return u8

##############################################################################################
class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
    
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x
    
##############################################################################################
# Discriminator: Patch Gan 기반 
# Patch Gan: 이미지를 16x16의 패치로 분할하여 각 패치가 진짜인지 가짜인지 식별
# high-frequency에서 정확도 향상

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        ### (256, 256) size의 이미지를 받아서, (16,16) 크기로 분할된 부분 이미지 16개가 출력될 수 있도록 shape을 잘 맞춰서 설계 해야 함. ###
        self.stage_1 = Dis_block(in_channels*2, out_channels=64, normalize=True)
        self.stage_2 = Dis_block(64,128)
        self.stage_3 = Dis_block(128,256)
        self.stage_4 = Dis_block(256,512)

        self.patch = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3,padding=1) # 16x16 패치 생성 <= Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #print(f"self.stage_1: {self.stage_1}")
        #print(f"self.patch: {self.patch}")

    def forward(self, a, b):          # a: torch.Size([B, 3, 256, 256]) , b: torch.Size([16, 3, 256, 256]) <= [B, C, H, W]
        x = torch.cat((a,b),dim=1)  # torch.Size([B, 6, 256, 256]) <= [B, (C+C), H, W]
        x = self.stage_1(x)     # torch.Size([B, 64, 128, 128]) <= self.stage_1에서 출력되는 out_channels=64이고, kernel=3, stride=2, padding=1 임으로 [B, 64, 128, 128] 출력
        x = self.stage_2(x)     # torch.Size([B, 128, 64, 64])  <= self.stage_2에서 출력되는 out_channels=128이고, kernel=3, stride=2, padding=1 임으로 [B, 128, 64, 64] 출력
        x = self.stage_3(x)     # torch.Size([B, 256, 32, 32])  <= self.stage_3에서 출력되는 out_channels=256이고, kernel=3, stride=2, padding=1 임으로 [B, 256, 32, 32] 출력
        x = self.stage_4(x)     # torch.Size([B, 512, 16, 16])  <= self.stage_4에서 출력되는 out_channels=512이고, kernel=3, stride=2, padding=1 임으로 [B, 512, 16, 16] 출력
        x = self.patch(x)       # torch.Size([B, 1, 16, 16])    <= self.patch에서 출력되는 out_channels=1이고, kernel=3, stride=1, padding=1 임으로 [B, 1, 16, 16] 출력
        x = torch.sigmoid(x)    # torch.Size([B, 1, 16, 16])    <= torch.sigmoid()는 입력 tensor의 shape와 같은 shape의 tensor 출력
        return x

