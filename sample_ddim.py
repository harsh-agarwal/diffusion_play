import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from src import DiffusionWrapper, device

class DDIMSampler:
    def __init__(
        self,
        n_steps=1000,
        n_inference_steps=50,  # DDIM can use much fewer steps
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    ):
        """
        DDIM Sampler implementation
        Args:
            n_steps: Number of steps used in training
            n_inference_steps: Number of steps to use during inference (can be much less than n_steps)
            beta_start, beta_end: noise schedule parameters
        """
        self.n_steps = n_steps
        self.n_inference_steps = n_inference_steps
        self.device = device
        
        # Original diffusion parameters (from training)
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Select subset of timesteps for inference (from most noisy to least noisy)
        self.timesteps = torch.linspace(n_steps - 1, 0, n_inference_steps).long()
        
    def ddim_step(
        self,
        model,
        x_t,
        t,
        t_prev,
        eta=0.0  # η=0 for DDIM, η=1 recovers DDPM
    ):
        """Single DDIM step"""
        # Get alpha values for current and previous timesteps
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0).to(self.device)
        
        # Predict noise
        eps = model(x_t, t.expand(x_t.shape[0]))
        
        # DDIM reverse process
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        sqrt_alphas_cumprod_t = torch.sqrt(alpha_cumprod_t)
        
        # Predict x_0
        pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * eps) / sqrt_alphas_cumprod_t
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clamp to ensure valid images
        
        # Direction pointing to x_t
        sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
        
        # Random noise for stochasticity
        noise = torch.randn_like(x_t) if eta > 0 else 0
        
        # Compute the next sample
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + \
                torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * eps + \
                sigma_t * noise
        
        return x_prev
    
    @torch.no_grad()
    def sample(self, model, n_samples=16, eta=0.0):
        """
        Generate samples using DDIM
        Args:
            model: The diffusion model
            n_samples: Number of samples to generate
            eta: Controls the stochasticity (0 = deterministic DDIM, 1 = DDPM-like)
        """
        model.eval()
        
        # Start from pure noise
        x = torch.randn(n_samples, 1, 28, 28).to(self.device)
        
        # Progressively denoise
        for i in range(len(self.timesteps) - 1):
            t = self.timesteps[i].to(self.device)
            t_prev = self.timesteps[i + 1].to(self.device)
            
            x = self.ddim_step(
                model=model,
                x_t=x,
                t=t,
                t_prev=t_prev,
                eta=eta
            )
            
        return x

def generate_and_save_samples(
    diffusion_model,
    n_samples=16,
    n_inference_steps=50,
    eta=0.0,
    save_path='ddim_samples.png'
):
    """Generate and visualize samples using DDIM"""
    # Initialize sampler
    sampler = DDIMSampler(n_inference_steps=n_inference_steps)
    
    # Generate samples
    samples = sampler.sample(
        model=diffusion_model.get_model(),
        n_samples=n_samples,
        eta=eta
    )
    
    # Denormalize samples
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)  # Ensure valid image range
    
    # Plot samples
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'DDIM Samples (steps={n_inference_steps}, eta={eta})')
    plt.savefig(save_path)
    plt.close()
    
    return samples

if __name__ == "__main__":
    print("Loading pre-trained model...")
    diffusion_model = DiffusionWrapper(device)
    checkpoint = torch.load('./diffusion_model_final.pt', map_location=device)
    diffusion_model.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate samples with different settings
    print("Generating samples with different settings...")
    
    # Pure DDIM (deterministic)
    generate_and_save_samples(
        diffusion_model,
        n_inference_steps=50,
        eta=0.0,
        save_path='ddim_samples_deterministic.png'
    )
    
    # Slightly stochastic
    generate_and_save_samples(
        diffusion_model,
        n_inference_steps=50,
        eta=0.3,
        save_path='ddim_samples_stochastic.png'
    )

    # replicate DDPM 
    generate_and_save_samples(
        diffusion_model,
        n_inference_steps=1000,
        eta=1.0,
        save_path='ddpm_samples_via_ddim.png'
    )
    
    # Compare different number of steps
    for steps in [20, 50, 100, 200, 500]:
        generate_and_save_samples(
            diffusion_model,
            n_inference_steps=steps,
            eta=0.0,
            save_path=f'ddim_samples_{steps}_steps.png'
        )
    
    print("Done! Check the generated PNG files for results.") 