import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + shortcut)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(b, c, -1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        
        attn = torch.einsum('bci,bcj->bij', q, k) / (self.channels ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(b, c, h, w)
        return x + self.proj(out)

class SingleStepUNet(nn.Module):
    """
    Modified UNet that predicts clean images in a single step
    Takes noisy image and noise level as input, outputs clean image directly
    """
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()
        
        # Store time_dim as instance variable
        self.time_dim = time_dim
        
        # Time embedding with more capacity for single-step prediction
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim * 4),  # Increased capacity
        )
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, 64, 7, padding=3)
        
        # Encoder
        self.down1 = nn.ModuleList([
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            AttentionBlock(128),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
        ])
        
        self.down2 = nn.ModuleList([
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            AttentionBlock(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
        ])
        
        self.down3 = nn.ModuleList([
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
        ])
        
        # Bottleneck with increased capacity
        self.bottleneck = nn.ModuleList([
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
        ])
        
        # Time embedding projections with proper dimensions
        self.time_proj1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim * 4, 512),  # Project to channel dimension
        )
        
        self.time_proj2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim * 4, 512),  # Project to channel dimension
        )
        
        self.time_proj3 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim * 4, 256),  # Project to channel dimension
        )
        
        # Decoder with skip connections and more capacity
        self.up1 = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            ResidualBlock(768, 256),  # 512 from skip connection
            AttentionBlock(256),
            ResidualBlock(256, 256),
        ])
        
        self.up2 = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            ResidualBlock(384, 128),  # 256 from skip connection
            AttentionBlock(128),
            ResidualBlock(128, 128),
        ])
        
        self.up3 = nn.ModuleList([
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            ResidualBlock(192, 64),  # 128 from skip connection
            AttentionBlock(64),
            ResidualBlock(64, 64),
        ])
        
        # Output layers with residual connection
        self.final = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
        )
        
        # Direct skip connection from input to output
        self.skip_scaling = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, t):
        # Initial time embedding
        t = t.unsqueeze(-1).float()  # [B, 1]
        t = self.time_mlp(t)         # [B, time_dim * 4]
        
        # Store input for final skip connection
        input_x = x
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skips = []
        
        # Down 1
        for layer in self.down1[:-1]:
            x = layer(x)
        skips.append(x)
        x = self.down1[-1](x)
        
        # Down 2
        for layer in self.down2[:-1]:
            x = layer(x)
        skips.append(x)
        x = self.down2[-1](x)
        
        # Down 3
        for layer in self.down3[:-1]:
            x = layer(x)
        skips.append(x)
        x = self.down3[-1](x)
        
        # Bottleneck with time embedding
        for i, layer in enumerate(self.bottleneck):
            x = layer(x)
            if i == len(self.bottleneck) // 2:
                # Project time embedding and add to features
                time_emb = self.time_proj1(t)  # [B, 512]
                time_emb = time_emb.view(x.shape[0], -1, 1, 1)  # [B, 512, 1, 1]
                time_emb = time_emb.expand(-1, -1, x.shape[2], x.shape[3])  # [B, 512, H, W]
                x = x + time_emb
        
        # Decoder with time-conditioned skip connections
        # Up 1
        x = self.up1[0](x)
        skip_x = skips.pop()
        # Add time embedding to skip connection
        time_emb = self.time_proj2(t)  # [B, 256]
        time_emb = time_emb.view(skip_x.shape[0], -1, 1, 1)  # [B, 256, 1, 1]
        time_emb = time_emb.expand(-1, -1, skip_x.shape[2], skip_x.shape[3])  # [B, 256, H, W]
        skip_x = skip_x + time_emb
        # Pad x on top and left to match skip_x dimensions
        pad_h = skip_x.shape[2] - x.shape[2]
        pad_w = skip_x.shape[3] - x.shape[3]
        x = F.pad(x, (pad_w, 0, pad_h, 0), mode='constant', value=0)
        x = torch.cat([x, skip_x], dim=1)
        for layer in self.up1[1:]:
            x = layer(x)
        
        # Up 2
        x = self.up2[0](x)
        skip_x = skips.pop()
        # Add time embedding to skip connection
        time_emb = self.time_proj3(t)  # [B, 128]
        time_emb = time_emb.view(skip_x.shape[0], -1, 1, 1)  # [B, 128, 1, 1]
        time_emb = time_emb.expand(-1, -1, skip_x.shape[2], skip_x.shape[3])  # [B, 128, H, W]
        skip_x = skip_x + time_emb
        x = torch.cat([x, skip_x], dim=1)
        for layer in self.up2[1:]:
            x = layer(x)
        
        # Up 3
        x = self.up3[0](x)
        skip_x = skips.pop()
        x = torch.cat([x, skip_x], dim=1)
        for layer in self.up3[1:]:
            x = layer(x)
        
        # Final layers with learned skip connection from input
        x = self.final(x)
        x = x + input_x * self.skip_scaling.exp()
        
        return x 