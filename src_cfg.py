import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from src import DiffusionWrapper, device, n_steps, alphas, alphas_cumprod, betas

# Simple CNN Classifier for MNIST
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_classifier(classifier, train_loader, num_epochs=5):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    classifier.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    return classifier

@torch.enable_grad()
def get_classifier_gradients(x_t, t, classifier, target_class):
    """Calculate classifier gradients for guidance"""
    x_t.requires_grad_(True)
    
    # Get classifier prediction
    logits = classifier(x_t)
    
    # Calculate log probability of target class
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs[:, target_class]
    
    # Calculate gradients
    gradients = torch.autograd.grad(selected.sum(), x_t)[0]
    
    return gradients

@torch.no_grad()
def sample_with_classifier_guidance(
    diffusion_model,
    classifier,
    target_class,
    n_samples=16,
    guidance_scale=3.0
):
    """Sample from diffusion model with classifier guidance"""
    model = diffusion_model.get_model()
    model.eval()
    classifier.eval()
    
    # Start from random noise
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    
    # Reverse diffusion process with classifier guidance
    for t in reversed(range(n_steps)):
        t_batch = torch.ones(n_samples).to(device) * t
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Get classifier gradients
        with torch.enable_grad():
            grad = get_classifier_gradients(x, t_batch, classifier, target_class)
        
        # Modify the predicted noise using classifier gradients
        predicted_noise = predicted_noise - guidance_scale * grad
        
        # Single diffusion step
        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
    
    return x

def generate_class_samples(
    diffusion_model,
    classifier,
    target_class,
    n_samples=16,
    guidance_scale=3.0
):
    """Generate and visualize samples for a specific class"""
    samples = sample_with_classifier_guidance(
        diffusion_model,
        classifier,
        target_class,
        n_samples,
        guidance_scale
    )
    
    # Denormalize samples
    samples = (samples + 1) / 2
    
    # Plot samples
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Generated samples for class {target_class}')
    plt.savefig(f'samples_class_{target_class}.png')
    plt.close()
    
    return samples

if __name__ == "__main__":
    # Load your trained diffusion model
    diffusion_model = DiffusionWrapper(device)
    checkpoint = torch.load('./diffusion_model_final.pt', map_location=device)
    diffusion_model.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize and train classifier
    from torch.utils.data import DataLoader
    from torchvision import datasets
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    classifier = MNISTClassifier().to(device)
    classifier = train_classifier(classifier, train_loader)
    
    # Generate samples for each digit
    for digit in range(10):
        generate_class_samples(
            diffusion_model,
            classifier,
            target_class=digit,
            n_samples=16,
            guidance_scale=8.0
        ) 