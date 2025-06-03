import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src import DiffusionWrapper, device, n_steps

class ConsistencyModel(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        
        # Time projection for upsampling path
        self.time_proj = nn.ModuleDict({
            'up1': nn.Linear(128, base_channels * 4),
            'up2': nn.Linear(128, base_channels),
            'final': nn.Linear(128, base_channels),
        })
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down1 = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
        ])
        
        self.down2 = nn.ModuleList([
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
        ])
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
        ])
        
        # Decoder with time embedding
        self.up1 = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
        ])
        
        self.up2 = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        ])
        
        # Output layers
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )
        
    def forward(self, x, t):
        # Time embedding
        
        t = t.unsqueeze(-1).float()
        t = self.time_mlp(t)  # [B, 128]
        
        # Initial conv
        hidden = self.init_conv(x)
        
        # Downsample
        for layer in self.down1:
            hidden = layer(hidden)
        skip1 = hidden
        
        for layer in self.down2:
            hidden = layer(hidden)
        skip2 = hidden
        
        # Bottleneck
        for layer in self.bottleneck:
            hidden = layer(hidden)
        
        # Upsample with time embedding
        # First upsampling block
        hidden = hidden + skip2
        hidden = self.up1[0](hidden)  # Transposed conv
        hidden = self.up1[1](hidden)  # GroupNorm
        hidden = self.up1[2](hidden)  # GELU
        hidden = self.up1[3](hidden)  # Conv
        time_emb = self.time_proj['up1'](t)
        time_emb = time_emb.view(-1, 1, 16, 16)
        time_emb = F.interpolate(time_emb, size=(hidden.shape[2], hidden.shape[3]), mode='bilinear', align_corners=False)
        time_emb = time_emb.repeat(1, hidden.shape[1], 1, 1)
        hidden = hidden + time_emb
        # Second upsampling block
        hidden = hidden + skip1
        hidden = self.up2[0](hidden)  # Transposed conv
        hidden = self.up2[1](hidden)  # GroupNorm
        hidden = self.up2[2](hidden)  # GELU
        hidden = self.up2[3](hidden)  # Conv
        
        
        time_emb = self.time_proj['up2'](t)
        time_emb = time_emb.view(-1, 1, 8, 8)
        time_emb = F.interpolate(time_emb, size=(hidden.shape[2], hidden.shape[3]), mode='bilinear', align_corners=False)
        time_emb = time_emb.repeat(1, hidden.shape[1], 1, 1)
        
        hidden = hidden + time_emb

        # hidden = hidden + time_emb.unsqueeze(-1).unsqueeze(-1)
        # hidden = hidden + skip1  # Skip connection
        
        # Final processing with time embedding
        hidden = self.final[0](hidden)  # GroupNorm
        hidden = self.final[1](hidden)  # GELU
        time_emb = self.time_proj['final'](t)
        time_emb = time_emb.view(-1, 1, 8, 8)
        time_emb = F.interpolate(time_emb, size=(hidden.shape[2], hidden.shape[3]), mode='bilinear', align_corners=False)
        time_emb = time_emb.repeat(1, hidden.shape[1], 1, 1)
        
        hidden = hidden + time_emb
        hidden = self.final[2](hidden)  # Final conv
        
        return hidden

class ConsistencyTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        n_steps=1000,
        sigma_min=0.002,
        sigma_max=1.0,
        rho=7.0,
        device=device
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.device = device
        self.n_steps = n_steps
        
        # Noise schedule parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        
        # Calculate sigmas
        self.sigmas = self.get_sigmas()
        
    def get_sigmas(self):
        """Get noise levels following the paper's karras schedule"""
        ramp = torch.linspace(0, 1, self.n_steps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas.to(self.device)
    
    def add_noise(self, x, noise_level):
        """Add noise to images"""
        noise = torch.randn_like(x)
        noisy = x + noise_level.view(-1, 1, 1, 1) * noise
        return noisy, noise
    
    def consistency_loss(self, x, t1, t2):
        """Compute consistency loss between teacher and student predictions"""
        # Get noise levels
        sigma1 = self.sigmas[t1]
        sigma2 = self.sigmas[t2]
        
        # Add noise to images
        x1, noise1 = self.add_noise(x, sigma1)
        x2, noise2 = self.add_noise(x, sigma2)
        
        # Get teacher predictions (with no grad)
        with torch.no_grad():
            # Teacher predicts noise
            teacher_noise1 = self.teacher(x1, t1)
            teacher_noise2 = self.teacher(x2, t2)
            
            # Convert teacher's noise prediction to image prediction
            teacher_pred1 = (x1 - sigma1.view(-1, 1, 1, 1) * teacher_noise1) / (1 + sigma1.view(-1, 1, 1, 1))
            teacher_pred2 = (x2 - sigma2.view(-1, 1, 1, 1) * teacher_noise2) / (1 + sigma2.view(-1, 1, 1, 1))
        
        # Get student predictions (directly predicts clean image)
        student_pred1 = self.student(x1, t1)
        student_pred2 = self.student(x2, t2)
        
        # Compute losses:
        # 1. Student should match teacher's predictions
        teacher_loss = (F.mse_loss(student_pred1, teacher_pred1) + 
                       F.mse_loss(student_pred2, teacher_pred2)) / 2.0
        
        # 2. Student predictions should be consistent
        consistency_loss = F.mse_loss(student_pred1, student_pred2)
        
        # 3. Student should predict clean images
        clean_loss = (F.mse_loss(student_pred1, x) + 
                     F.mse_loss(student_pred2, x)) / 2.0
        
        # Combine losses with weights
        total_loss = (
            0.7 * teacher_loss +    # Match teacher
            0.3 * consistency_loss + # Be consistent
            0.0 * clean_loss        # made clean loss zero to dbug things better 
        )
        
        return total_loss
    
    def train(
        self,
        train_loader,
        n_epochs,
        learning_rate=1e-4,
        save_interval=500,
        log_dir='runs/consistency'
    ):
        optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        writer = SummaryWriter(log_dir)
        
        for epoch in range(n_epochs):
            self.student.train()
            epoch_loss = 0
            
            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                
                # Sample two random timesteps
                t1 = torch.randint(0, self.n_steps, (images.shape[0],)).to(self.device)
                t2 = torch.randint(0, self.n_steps, (images.shape[0],)).to(self.device)
                
                # Compute loss
                loss = self.consistency_loss(images, t1, t2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
                    # Log samples
                    if batch_idx % 500 == 0:
                        with torch.no_grad():
                            self.student.eval()
                            # Generate samples
                            noise = torch.randn_like(images[:8])
                            samples = self.student(noise, torch.zeros(8).to(self.device))
                            writer.add_images('Generated', samples, epoch * len(train_loader) + batch_idx)
                            self.student.train()
            
            avg_loss = epoch_loss / len(train_loader)
            writer.add_scalar('Loss/epoch', avg_loss, epoch)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, f'consistency_model_epoch_{epoch+1}.pt')
        
        # Save final model
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, 'consistency_model_final.pt')
        
        writer.close()
        return self.student

def train_consistency_model():
    # Load the pre-trained diffusion model
    print("Loading teacher model...")
    teacher_wrapper = DiffusionWrapper(device)
    checkpoint = torch.load('./train_runs_diffusion/diffusion_model_final.pt', map_location=device)
    teacher_wrapper.model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model = teacher_wrapper.get_model()
    teacher_model.eval()
    
    # Create student model
    print("Creating student model...")
    student_model = ConsistencyModel().to(device)
    
    # Data loading
    print("Setting up data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = ConsistencyTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        device=device
    )
    
    # Train the model
    print("Starting consistency training...")
    trainer.train(
        train_loader=train_loader,
        n_epochs=5000,
        learning_rate=1e-4,
        save_interval=5
    )

if __name__ == "__main__":
    train_consistency_model() 