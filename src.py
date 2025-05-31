import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models import DiffusionWrapper

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
n_steps = 1000
beta_start = 0.0001
beta_end = 0.02
batch_size = 128
n_epochs = 100
learning_rate = 2e-4

# Calculate diffusion parameters
betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

def get_noisy_image(x_start, t):
    """Add noise to image at timestep t"""
    noise = torch.randn_like(x_start)
    return (
        sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_start +
        sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise,
        noise
    )

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
diffusion_model = DiffusionWrapper(device)
model = diffusion_model.get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train():
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/diffusion_model')
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, n_steps, (images.shape[0],)).to(device)
            
            # Get noisy images and noise
            noisy_images, noise = get_noisy_image(images, t)
            
            # Predict noise
            predicted_noise = model(noisy_images, t)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log batch loss
            global_step = epoch * len(dataloader) + batch
            writer.add_scalar('Loss/batch', loss.item(), global_step)
            
            if batch % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch}, Loss: {loss.item():.4f}")
                
                # Log images periodically
                if batch % 500 == 0:
                    # Log original images
                    writer.add_images('Original', images[:8], global_step)
                    # Log noisy images
                    writer.add_images('Noisy', noisy_images[:8], global_step)
                    # Generate and log samples
                    with torch.no_grad():
                        model.eval()
                        samples = sample(8)
                        writer.add_images('Generated', samples[:8], global_step)
                        model.train()
        
        # Log epoch average loss
        avg_loss = epoch_loss/len(dataloader)
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch when a multiple of 5
        if epoch % 5 == 0:
            diffusion_model.save_model(
                f'diffusion_model_epoch_{epoch+1}.pt',
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss
            )
            
            # Log model graph once
            if epoch == 0:
                sample_images = images[:2].to(device)
                sample_t = torch.zeros(2).to(device)
                writer.add_graph(model, (sample_images, sample_t))
    
    # Save final model
    diffusion_model.save_model(
        'diffusion_model_final.pt',
        optimizer=optimizer,
        epoch=n_epochs-1,
        loss=avg_loss
    )
    
    writer.close()
    return model

# Sampling
@torch.no_grad()
def sample(n_samples=16):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    
    for t in reversed(range(n_steps)):
        t_batch = torch.ones(n_samples).to(device) * t
        predicted_noise = model(x, t_batch)
        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
    
    return x

if __name__ == "__main__":
    print("Starting training...")
    trained_model = train()
    
    print("Generating samples...")
    samples = sample()  # Uses the trained model since it's global
    samples = (samples + 1) / 2  # Denormalize
    
    # Plot samples
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.savefig('samples.png')
    plt.close()
