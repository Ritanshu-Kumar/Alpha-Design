import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

class OptimizerManager:
    def __init__(self, network, learning_rate=1e-3, weight_decay=1e-4):
        self.network = network
        self.lr = learning_rate
        self.weight_decay = weight_decay
        
        self.optimizer = optim.AdamW(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )   
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=50, 
            factor=0.5
        )

    def use_cosine_warm(self, t0=10, t_mul=2, eta_min=1e-6):
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=t0,
            T_mult=t_mul,
            eta_min=eta_min
        )
        print(f"🔄 Switched to Cosine Warm Restarts (T₀={t0}, T_mult={t_mul})")

    def use_adamw_cosine(self, t0=10, t_mult=2, eta_min=1e-6, 
                         lr=2e-4, weight_decay=1e-3):
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=t0, 
            T_mult=t_mult,
            eta_min=eta_min
        )
        print(f"⚡ Enhanced optimizer: AdamW + Cosine (lr={lr}, wd={weight_decay})")
    
    def get_optimizer(self):
        return self.optimizer
    
    def update_learning_rate(self, cfd_loss):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(cfd_loss)
        else:
            pass
    
    def step_cosine_scheduler(self, epoch, batch_idx=0, total_batches=1):
        if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step(epoch + batch_idx / total_batches)
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def separate_head_optimizers(self, policy_lr=2e-4, value_lr=2e-4):
        policy_optimizer = optim.AdamW(
            self.network.policy_head.parameters(),
            lr=policy_lr,
            weight_decay=self.weight_decay
        )
        
        value_optimizer = optim.AdamW(
            self.network.value_head.parameters(),
            lr=value_lr,
            weight_decay=self.weight_decay
        )
        
        return policy_optimizer, value_optimizer
    
    def reset_optimizer_state(self):
        self.optimizer.state = {}