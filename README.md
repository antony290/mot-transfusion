
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
