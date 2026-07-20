
# 1. SSH连接服务器 (替换成您的实际信息)
ssh username@your_server_ip

# 2. 克隆仓库
cd /mnt/data/zhouheng
git clone https://github.com/antony290/mot-transfusion.git
cd mot-transfusion

# 3. 创建/激活Python环境 (推荐用conda)
conda activate base  # 或您的环境名

# 4. 安装依赖
pip install torch torchvision einops accelerate beartype jaxtyping ema-pytorch adam-atan2-pytorch axial-positional-embedding rotary-embedding-torch torchdiffeq loguru tqdm datasets pycocotools

# 5. 运行训练
python train_mot_multi_modality.py

###下载数据集

# 设置HuggingFace镜像（国内服务器推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 再运行下载
python -c "
from datasets import load_dataset
ds = load_dataset('lmms-lab/coco-captions', cache_dir='/mnt/data/zhouheng')
print(f'下载完成，样本数: {len(ds[\"train\"])}')
"