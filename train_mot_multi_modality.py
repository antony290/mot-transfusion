"""
基于 Mixture-of-Transformers (MoT) 的双模态训练脚本
- 数据集: COCO Captions (真正的多模态: 图像 + 文本描述)
- 模态1: 文本 (COCO caption 英文句子, 字符级编码)
- 模态2: 图像 (RGB 图像, 通过编码器映射到latent空间)
- 训练硬件: NVIDIA RTX 5090 (32GB显存)
- 参考论文: https://arxiv.org/pdf/2411.04996
"""

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
from PIL import Image

from transfusion_pytorch import Transfusion, print_modality_sample

from accelerate import Accelerator
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ============ 路径配置 (远程Linux服务器) ============
DATA_CACHE_DIR = "/mnt/data/zhouheng"  # 数据集缓存目录

rmtree('./results_mot', ignore_errors=True)
results_folder = Path('./results_mot')
results_folder.mkdir(exist_ok=True, parents=True)


def divisible_by(num, den):
    return (num % den) == 0


# ============ 图像编码器/解码器 (适配RGB 3通道) ============
class ImageEncoder(Module):
    """将RGB图像编码到latent空间
    输入: (B, 3, H, W) 范围[0, 1]
    输出: (B, dim_latent, H/4, W/4)
    两次下采样: 64x64 -> 32x32 -> 16x16
    """
    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent = dim_latent
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim_latent, 4, stride=2, padding=1),  # RGB 3通道输入
            nn.SiLU(),
            nn.Conv2d(dim_latent, dim_latent, 4, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        x = x * 2 - 1  # 归一化到[-1, 1]
        x = self.conv(x)
        return x


class ImageDecoder(Module):
    """将latent解码回RGB图像
    输入: (B, dim_latent, H/4, W/4)
    输出: (B, 3, H, W) 范围[0, 1]
    """
    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent = dim_latent
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dim_latent, dim_latent, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(dim_latent, 3, 4, stride=2, padding=1),  # RGB 3通道输出
        )

    def forward(self, x):
        x = self.deconv(x)
        x = (x + 1) * 0.5
        return x.clamp(min=0., max=1.)


# ============ COCO Captions 多模态数据集 ============
IMAGE_SIZE = 64  # 图像resize到64x64, 经两次下采样后latent为16x16
MAX_TEXT_LEN = 96  # caption最大字符长度

class COCOMultiModalDataset(Dataset):
    """COCO Captions 数据集
    - 每个样本包含: 图像 + 对应的文本描述(caption)
    - 图像: RGB, resize到64x64
    - 文本: 英文caption, 字符级编码(ASCII)
    """
    def __init__(self, cache_dir=DATA_CACHE_DIR, image_size=IMAGE_SIZE, split="train"):
        super().__init__()
        from datasets import load_dataset

        print(f"正在加载COCO Captions数据集, 缓存目录: {cache_dir}")
        # 使用HuggingFace datasets加载COCO captions
        self.dataset = load_dataset(
            "lmms-lab/coco-captions",
            split=split,
            cache_dir=cache_dir,
        )
        print(f"数据集加载完成, 共 {len(self.dataset)} 个样本")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # --- 图像模态 ---
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.transform(image)  # (3, 64, 64), [0, 1]

        # --- 文本模态 (真实的caption描述) ---
        # COCO captions: 每张图有多个caption, 随机选一个
        captions = item.get('captions', item.get('caption', []))
        if isinstance(captions, list):
            caption = random.choice(captions) if len(captions) > 0 else ""
        else:
            caption = captions

        # 字符级编码 (ASCII, 范围0-127)
        text_str = caption[:MAX_TEXT_LEN]
        text_tokens = tensor([ord(c) for c in text_str], dtype=torch.long)

        return [text_tokens, image_tensor]


def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch


# ============ MoT 模型配置 ============
model = Transfusion(
    num_text_tokens=256,          # ASCII字符空间
    dim_latent=128,               # latent维度
    channel_first_latent=True,    # latent为通道优先格式
    modality_default_shape=(16, 16),  # 图像latent空间形状 (64/4=16)
    modality_num_dim=2,           # 2D图像
    add_pos_emb=True,             # 添加位置编码
    modality_encoder=ImageEncoder(128),
    modality_decoder=ImageDecoder(128),
    velocity_consistency_loss_weight=0.1,
    reconstruction_loss_weight=0.1,
    model_output_clean=True,
    use_mot=True,                 # 启用 Mixture-of-Transformers
    transformer=dict(
        dim=768,
        depth=12,
        dim_head=64,
        heads=12,
        attn_laser=True,
        ff_expansion_factor=4,
    )
)

print(f"MoT模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


# ============ 训练主流程 ============
if __name__ == '__main__':
    ema_model = model.create_ema()

    dataset = COCOMultiModalDataset(
        cache_dir=DATA_CACHE_DIR,
        image_size=IMAGE_SIZE,
        split="train",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=None,  # 使用默认collate (会自动padding)
    )
    iter_dl = cycle(dataloader)

    optimizer = MuonAdamAtan2(
        model.muon_parameters(),
        model.parameters(),
        lr=8e-4,
    )

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=2,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    ema_model.to(accelerator.device)

    print(f"训练设备: {accelerator.device}")
    print(f"混合精度: bf16, 梯度累积: 2步")
    print(f"开始训练...")

    for step in range(1, 200_000 + 1):
        batch = next(iter_dl)

        loss = model(batch, velocity_consistency_ema_model=ema_model)

        accelerator.backward(loss)

        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        optimizer.zero_grad()

        ema_model.update()

        if step % 100 == 0:
            accelerator.print(f"step {step}: loss = {loss.item():.3f}")

        # 定期生成样本验证
        if divisible_by(step, 1000):
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                model.eval()
                with torch.no_grad():
                    image = ema_model.generate_modality_only(batch_size=16)

                    save_image(
                        rearrange(image, '(gh gw) c h w -> c (gh h) (gw w)', gh=4).detach().cpu(),
                        str(results_folder / f'step_{step}.png')
                    )
                model.train()
                print(f"已保存生成样本: {results_folder / f'step_{step}.png'}")

        # 定期保存checkpoint
        if divisible_by(step, 5000):
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, str(results_folder / f'checkpoint_{step}.pt'))
                print(f"已保存checkpoint: checkpoint_{step}.pt")
