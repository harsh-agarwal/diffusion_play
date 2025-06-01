import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from distill_model import SingleStepUNet
from trainer import DistillationTrainer
import sys
sys.path.append('..')
from models import DiffusionWrapper
import glob 

# Check if MPS is available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def get_latest_checkpoint():
    """Find the latest checkpoint in the checkpoints directory"""
    checkpoints = glob.glob('checkpoints/distilled_model_epoch_*.pt')
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the latest
    epochs = [int(ckpt.split('_epoch_')[-1].split('.')[0]) for ckpt in checkpoints]
    latest_idx = epochs.index(max(epochs))
    return checkpoints[latest_idx]



def main():
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Load the teacher model
    print("Loading teacher model...")
    teacher_wrapper = DiffusionWrapper(device)
    teacher_wrapper.load_model('./train_runs_diffusion/diffusion_model_final.pt')  # Load your trained model
    teacher_model = teacher_wrapper.get_model()
    teacher_model.eval()  # Set to eval mode
    
    # Create student model
    print("Creating student model...")
    student_model = SingleStepUNet().to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    start_epoch = 0
    
    # Check for existing checkpoint
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")


    # Data loading
    print("Setting up data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=256,  # Smaller batch size for better stability
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        device=device,
        n_steps=1000,  # Same as teacher
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Train the student model
    print("Starting distillation training...")
    student_model = trainer.train(
        train_dataloader=dataloader,
        n_epochs=100,  # More epochs for better convergence
        learning_rate=1e-4,
        save_path='checkpoints/distilled_model',
        log_interval=50,
        save_interval=10,
        start_epoch=start_epoch,
        optimizer=optimizer
    )
    
    # Generate comparison samples
    print("Generating comparison samples...")
    n_samples = 16
    
    # Generate samples from both models
    with torch.no_grad():
        # Random noise
        noise = torch.randn(n_samples, 1, 28, 28).to(device)
        t = torch.ones(n_samples, device=device) * (trainer.n_steps - 1)
        
        # Get predictions
        teacher_samples, _ = trainer.teacher_denoising(noise, t)
        student_samples = trainer.sample(n_samples)
        
        # Denormalize
        teacher_samples = (teacher_samples + 1) / 2
        student_samples = (student_samples + 1) / 2
        
        # Plot samples side by side
        plt.figure(figsize=(20, 10))
        
        # Teacher samples
        for i in range(n_samples):
            plt.subplot(4, 8, i + 1)
            plt.imshow(teacher_samples[i, 0].cpu().numpy(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Teacher (1000 steps)', pad=10)
        
        # Student samples
        for i in range(n_samples):
            plt.subplot(4, 8, i + n_samples + 1)
            plt.imshow(student_samples[i, 0].cpu().numpy(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Student (1 step)', pad=10)
        
        plt.tight_layout()
        plt.savefig('samples/comparison_samples.png')
        plt.close()
    
    print("Training complete! Check the samples directory for results.")

if __name__ == "__main__":
    main() 