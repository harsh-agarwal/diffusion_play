import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection handling
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut
        x = F.relu(x)
        return x

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=1, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Encoder
        self.down1 = nn.ModuleList([
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, 3, padding=1, stride=2)  # Downsample
        ])
        
        self.down2 = nn.ModuleList([
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 256, 3, padding=1, stride=2)  # Downsample
        ])
        
        self.down3 = nn.ModuleList([
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 512, 3, padding=1, stride=2)  # Downsample
        ])
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        ])
        
        # Time embedding projection
        self.time_proj = nn.Conv2d(time_dim, 512, 1)
        
        # Decoder
        self.up1 = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # Upsample
            ResidualBlock(512, 256),  # 512 because of skip connection
            ResidualBlock(256, 256),
        ])
        
        self.up2 = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # Upsample
            ResidualBlock(256, 128),  # 256 because of skip connection
            ResidualBlock(128, 128),
        ])
        
        self.up3 = nn.ModuleList([
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # Upsample
            ResidualBlock(128, 64),  # 128 because of skip connection
            ResidualBlock(64, 64),
        ])
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, in_channels, 3, padding=1)
        
        # Attention layers
        self.attention1 = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.SiLU(),
            nn.Conv2d(256, 256, 1)
        )
        
        self.attention2 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.SiLU(),
            nn.Conv2d(512, 512, 1)
        )
    
    def forward(self, x, t):
        # Time embedding
        
        t = t.unsqueeze(-1).float()
        t = self.time_mlp(t)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder
        skip1 = x
        for layer in self.down1[:-1]:
            x = layer(x)
        x = self.down1[-1](x)
        
        skip2 = x
        for layer in self.down2[:-1]:
            x = layer(x)
        x = self.down2[-1](x)
        
        skip3 = x
        x = self.attention1(x) + x  # Attention
        for layer in self.down3[:-1]:
            x = layer(x)
        x = self.down3[-1](x)
        
        # Bottleneck
        x = self.attention2(x) + x  # Attention
        for layer in self.bottleneck:
            x = layer(x)
            
        # Add time embedding
        t = t.unsqueeze(-1).unsqueeze(-1)
        t = t.repeat(1, 1, x.shape[2], x.shape[3])
        x = x + self.time_proj(t)
        
        # Decoder
        x = self.up1[0](x)
        x = F.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip3], dim=1)
        for layer in self.up1[1:]:
            x = layer(x)
            
        x = self.up2[0](x)
        x = torch.cat([x, skip2], dim=1)
        for layer in self.up2[1:]:
            x = layer(x)
            
        x = self.up3[0](x)
        x = torch.cat([x, skip1], dim=1)
        for layer in self.up3[1:]:
            x = layer(x)
            
        # Final conv
        x = self.final_conv(x)
        return x

class DiffusionWrapper:
    def __init__(self, device):
        self.model = ResNetUNet().to(device)
        self.device = device
    
    def get_model(self):
        return self.model
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model
    
    def save_model(self, path, optimizer=None, epoch=None, loss=None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        torch.save(checkpoint, path)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 256, 3, padding=1)
        
        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Conv2d(384, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, 1, 3, padding=1)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t = self.time_mlp(t)
        
        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = self.pool(x1)
        x2 = F.relu(self.enc2(x2))
        x3 = self.pool(x2)
        x3 = F.relu(self.enc3(x3))
        
        # Bottleneck
        x3 = self.bottleneck(x3)
        
        # Add time embedding
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x3.shape[2], x3.shape[3])
        x3 = x3 + t
        
        # Decoder
        x = self.upsample(x3)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.dec3(x))
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.dec2(x))
        
        x = self.dec1(x)
        return x 