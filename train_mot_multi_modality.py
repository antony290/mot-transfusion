from shutil import rmtree
from pathlib import Path
import random

import torch
from torch import tensor, nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from adam_atan2_pytorch import MuonAdamAtan2

from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from transfusion_pytorch import Transfusion, print_modality_sample

from accelerate import Accelerator

rmtree('./results_mot', ignore_errors = True)
results_folder = Path('./results_mot')
results_folder.mkdir(exist_ok = True, parents = True)

def divisible_by(num, den):
    return (num % den) == 0

class ImageEncoder(Module):
    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent = dim_latent
        self.conv = nn.Sequential(
            nn.Conv2d(1, dim_latent, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_latent, dim_latent, 4, stride=2, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = x * 2 - 1
        x = self.conv(x)
        return x

class ImageDecoder(Module):
    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent = dim_latent
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dim_latent, dim_latent, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim_latent, 1, 4, stride=2, padding=1),
        )
    
    def forward(self, x):
        x = self.deconv(x)
        x = (x + 1) * 0.5
        return x.clamp(min=0., max=1.)

digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

class MultiModalDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data',
            download = True,
            train = True
        )
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        pil, label = self.mnist[idx]
        image_tensor = T.PILToTensor()(pil).float() / 255
        
        text_str = digit_names[label]
        
        text_tokens = tensor([ord(c) for c in text_str], dtype=torch.long)
        
        return [text_tokens, image_tensor]

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = 128,
    channel_first_latent = True,
    modality_default_shape = (7, 7),
    modality_num_dim = 2,
    add_pos_emb = True,
    modality_encoder = ImageEncoder(128),
    modality_decoder = ImageDecoder(128),
    velocity_consistency_loss_weight = 0.1,
    reconstruction_loss_weight = 0.1,
    model_output_clean = True,
    use_mot = True,
    transformer = dict(
        dim = 512,
        depth = 8,
        dim_head = 64,
        heads = 8,
        attn_laser = True,
        ff_expansion_factor = 4,
    )
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

ema_model = model.create_ema()

dataset = MultiModalDataset()
dataloader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 4)
iter_dl = cycle(dataloader)

optimizer = MuonAdamAtan2(model.muon_parameters(), model.parameters(), lr = 8e-4)

accelerator = Accelerator(mixed_precision='bf16')

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
ema_model.to(accelerator.device)

for step in range(1, 200_000 + 1):
    batch = next(iter_dl)
    
    loss = model(batch, velocity_consistency_ema_model = ema_model)
    
    accelerator.backward(loss)
    
    accelerator.clip_grad_norm_(model.parameters(), 0.5)
    
    optimizer.step()
    optimizer.zero_grad()
    
    ema_model.update()
    
    if step % 100 == 0:
        accelerator.print(f'{step}: {loss.item():.3f}')
    
    if divisible_by(step, 1000):
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            image = ema_model.generate_modality_only(batch_size = 16)
            
            save_image(
                rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 4).detach().cpu(),
                str(results_folder / f'{step}.png')
            )