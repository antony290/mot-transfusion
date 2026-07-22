"""
基于 Mixture-of-Transformers (MoT) 的双模态训练脚本
- 数据集: COCO Captions (真正的多模态: 图像 + 文本描述)
- 模态1: 文本 (COCO caption 英文句子, 字符级编码)
- 模态2: 图像 (RGB 图像, 通过编码器映射到latent空间)
- 训练硬件: NVIDIA RTX 5090 (32GB显存)
- 参考论文: https://arxiv.org/pdf/2411.04996
"""

import os
# 设置HuggingFace镜像（国内服务器）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
DATA_CACHE_DIR = "/mnt/data/zhouheng/mot"  # 数据集缓存目录

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
    """COCO Captions 数据集 - 使用torchvision直接下载
    - 每个样本包含: 图像 + 对应的文本描述(caption)
    - 图像: RGB, resize到64x64
    - 文本: 英文caption, 字符级编码(ASCII)
    """
    def __init__(self, root=DATA_CACHE_DIR, image_size=IMAGE_SIZE, train=True):
        super().__init__()
        import torchvision.datasets as datasets
        
        print(f"正在加载COCO数据集, 根目录: {root}")
        
        # 使用torchvision加载COCO Captions
        img_folder = f"{root}/coco/train2017"
        ann_file = f"{root}/coco/annotations/captions_train2017.json"
        
        # 检查文件是否存在
        import os
        if not os.path.exists(img_folder):
            raise FileNotFoundError(f"图像文件夹不存在: {img_folder}\n请确保数据集已正确解压到 {root}/coco/")
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"标注文件不存在: {ann_file}\n请确保annotations/captions_train2017.json已正确放置")
        
        self.coco = datasets.CocoCaptions(
            root=img_folder,
            annFile=ann_file,
        )
        print(f"数据集加载完成, 共 {len(self.coco)} 个样本")

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        # torchvision返回: (image, [caption1, caption2, ...])
        image, captions = self.coco[idx]
        
        # --- 图像模态 ---
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.transform(image)  # (3, 64, 64), [0, 1]

        # --- 文本模态 (随机选一个caption) ---
        caption = random.choice(captions) if captions else ""

        # 字符级编码 (ASCII, 范围0-127)
        text_str = str(caption)[:MAX_TEXT_LEN]
        text_tokens = tensor([ord(c) for c in text_str], dtype=torch.long)

        return [text_tokens, image_tensor]


def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch


def collate_fn(batch):
    """自定义collate函数，处理变长文本和固定大小图像
    batch: list of [text_tokens, image_tensor]
    返回: list of [text_tokens_list, image_tensor_batch] 保持与Transfusion输入格式一致
    """
    text_list = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch], dim=0)
    
    # Transfusion期望的格式是 list of [text, image]，每个样本是一个列表
    # 我们需要将batch按样本组织，而不是按模态组织
    result = []
    for i in range(len(text_list)):
        result.append([text_list[i], images[i]])
    
    return result


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
        dim=512,
        depth=8,
        dim_head=64,
        heads=8,
        attn_laser=True,
        ff_expansion_factor=4,
    )
)

print(f"MoT模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


# ============ 评估函数 ============
def evaluate(model, dataloader, device, num_batches=50):
    """在验证集上计算平均loss"""
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            try:
                loss = model(batch)
                total_loss += loss.item()
                num_samples += 1
            except Exception as e:
                print(f"评估batch {i}出错: {e}")
                continue

    model.train()
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    return avg_loss


# ============ 训练主流程 ============
if __name__ == '__main__':
    ema_model = model.create_ema()

    # 加载完整数据集
    full_dataset = COCOMultiModalDataset(
        root=DATA_CACHE_DIR,
        image_size=IMAGE_SIZE,
        train=True,
    )

    # 划分训练集和验证集 (90% 训练, 10% 验证)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"数据集划分: 训练 {train_size} 样本, 验证 {val_size} 样本")

    # 训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # 验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    iter_dl = cycle(train_loader)

    optimizer = MuonAdamAtan2(
        model.muon_parameters(),
        model.parameters(),
        lr=8e-4,
    )

    # 确保编码器/解码器参数在优化器中
    all_params = set()
    for group in optimizer.param_groups:
        all_params.update(group['params'])
    
    encoder_params = list(model.modality_encoder[0].parameters())
    decoder_params = list(model.modality_decoder[0].parameters())
    
    missing_params = []
    for p in encoder_params + decoder_params:
        if p not in all_params:
            missing_params.append(p)
    
    if missing_params:
        print(f"警告: 添加编码器/解码器参数到优化器, 共 {len(missing_params)} 个参数")
        optimizer.add_param_group({'params': missing_params, 'lr': 8e-4})

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=4,
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    val_loader = accelerator.prepare(val_loader)
    ema_model.to(accelerator.device)

    # 训练记录
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"训练设备: {accelerator.device}")
    print(f"混合精度: bf16, 梯度累积: 2步")
    print(f"开始训练...")

    for step in range(1, 50_000 + 1):
        batch = next(iter_dl)

        loss = model(batch, velocity_consistency_ema_model=ema_model)

        accelerator.backward(loss)

        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        optimizer.zero_grad()

        ema_model.update()

        # 记录训练loss
        train_losses.append(loss.item())

        if step % 100 == 0:
            avg_train_loss = sum(train_losses[-100:]) / len(train_losses[-100:])
            accelerator.print(f"step {step}: train_loss = {loss.item():.3f}, avg_train_loss = {avg_train_loss:.3f}")

        # 定期验证评估
        if divisible_by(step, 500):
            accelerator.print(f"正在验证评估...")
            val_loss = evaluate(model, val_loader, accelerator.device, num_batches=20)
            val_losses.append(val_loss)

            accelerator.print(f"step {step}: val_loss = {val_loss:.3f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if accelerator.is_main_process:
                    torch.save({
                        'step': step,
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'val_loss': val_loss,
                    }, str(results_folder / 'best_model.pt'))
                    accelerator.print(f"保存最佳模型: val_loss = {val_loss:.3f}")

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
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                }, str(results_folder / f'checkpoint_{step}.pt'))
                print(f"已保存checkpoint: checkpoint_{step}.pt")

        # 定期保存训练曲线
        if divisible_by(step, 2000):
            if accelerator.is_main_process:
                import json
                with open(results_folder / 'training_log.json', 'w') as f:
                    json.dump({
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'best_val_loss': best_val_loss,
                    }, f)
                print(f"已保存训练日志")
