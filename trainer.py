import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class DistillationTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        device,
        n_steps=1000,
        beta_start=0.0001,
        beta_end=0.02
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.device = device
        
        # Setup noise schedule
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        self.n_steps = n_steps
        
    def get_noisy_image(self, x_start, t):
        """Add noise to image at timestep t"""
        noise = torch.randn_like(x_start)
        noisy_image = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_start +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise
        )
        return noisy_image, noise

    @torch.no_grad()
    def teacher_denoising(self, x_noisy, start_t):
        """Run the teacher model through multiple denoising steps"""
        self.teacher.eval()
        x = x_noisy.clone()
        
        # Store intermediate predictions for progressive distillation
        intermediate_preds = []
        
        for t in reversed(range(start_t.min().item() + 1)):
            t_batch = torch.ones(x.shape[0], device=self.device) * t
            
            # Get teacher's noise prediction
            predicted_noise = self.teacher(x, t_batch)
            
            # Store current prediction
            if t % (self.n_steps // 10) == 0:  # Store every 10% of steps
                intermediate_preds.append((t_batch, x.clone()))
            
            # Denoise step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
        
        return x, intermediate_preds
    
    def train_step(self, clean_images, optimizer):
        """Single training step for distillation"""
        batch_size = clean_images.shape[0]
        
        # Sample random timesteps (high noise levels)
        t = torch.randint(self.n_steps // 2, self.n_steps, (batch_size,)).to(self.device)
        
        # Get noisy images
        noisy_images, _ = self.get_noisy_image(clean_images, t)
        
        # Get teacher's multi-step predictions
        teacher_clean, intermediate_preds = self.teacher_denoising(noisy_images, t)
        
        # Student predicts clean image directly
        self.student.train()
        student_clean = self.student(noisy_images, t)
        
        # Main loss: match teacher's final clean prediction
        main_loss = F.mse_loss(student_clean, teacher_clean)
        
        # Progressive loss: also try to match intermediate predictions
        progressive_loss = 0.0
        for t_inter, x_inter in intermediate_preds:
            student_inter = self.student(x_inter, t_inter)
            progressive_loss += F.mse_loss(student_inter, teacher_clean)
        progressive_loss /= len(intermediate_preds)
        
        # Original image loss
        clean_loss = F.mse_loss(student_clean, clean_images)
        
        # Combined loss with weights
        total_loss = main_loss + 0.1 * progressive_loss + 0.1 * clean_loss
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'progressive_loss': progressive_loss.item(),
            'clean_loss': clean_loss.item()
        }
    
    def train(
        self,
        train_dataloader,
        n_epochs,
        learning_rate,
        save_path,
        log_interval=100,
        save_interval=5,
        start_epoch = 0,
        optimizer = None
    ):
        """Full training loop"""
        writer = SummaryWriter('runs/distillation')
        if optimizer is None:
            optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        
        for epoch in range(start_epoch, start_epoch + n_epochs):
            epoch_losses = {'total': 0, 'main': 0, 'progressive': 0, 'clean': 0}
            
            for batch_idx, (images, _) in enumerate(train_dataloader):
                images = images.to(self.device)
                
                # Training step
                losses = self.train_step(images, optimizer)
                for k, v in losses.items():
                    epoch_losses[k.split('_')[0]] += v
                
                # Logging
                if batch_idx % log_interval == 0:
                    print("i am here and i am writing into logs.")
                    print(f"Epoch {epoch+1}, Batch {batch_idx}")
                    for k, v in losses.items():
                        print(f"{k}: {v:.4f}")
                        writer.add_scalar(
                            f'Loss/{k}',
                            v,
                            epoch * len(train_dataloader) + batch_idx
                        )
                    
                    # Log sample images periodically
                    if batch_idx % (log_interval) == 0:
                        with torch.no_grad():
                            print("running evals to send into logs")
                            # Get a noisy version of current batch
                            t_vis = (torch.ones(8, device=self.device) * (self.n_steps - 1)).long()
                            noisy_vis, _ = self.get_noisy_image(images[:8], t_vis)
                            
                            # Get predictions
                            teacher_pred, _ = self.teacher_denoising(noisy_vis, t_vis)
                            student_pred = self.student(noisy_vis, t_vis)
                            
                            # Log images
                            writer.add_images('Original', images[:8], epoch)
                            writer.add_images('Teacher_Prediction', teacher_pred[:8], epoch)
                            writer.add_images('Student_Prediction', student_pred[:8], epoch)
            
            # Log epoch average losses
            for k, v in epoch_losses.items():
                avg_loss = v / len(train_dataloader)
                writer.add_scalar(f'Loss/epoch_{k}', avg_loss, epoch)
                print(f"Epoch {epoch+1} average {k} loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': epoch_losses,
                }, f"{save_path}_epoch_{epoch+1}.pt")
        
        # Save final model
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': epoch_losses,
        }, f"{save_path}_final.pt")
        
        writer.close()
        return self.student
    
    @torch.no_grad()
    def sample(self, n_samples=16):
        """Generate samples using the student model in one step"""
        self.student.eval()
        
        # Start from random noise
        x = torch.randn(n_samples, 1, 28, 28).to(self.device)
        
        # Set timestep to maximum (most noisy)
        t = torch.ones(n_samples).to(self.device) * (self.n_steps - 1)
        
        # Generate clean image in one step
        x = self.student(x, t)
        
        return x 