# config.py
from pathlib import Path
import torch
import random
import numpy as np


class Config:
    """
    项目的统一配置入口：
    - 路径（数据、日志、模型）
    - 图像预处理尺寸
    - 模型结构相关参数
    - 训练超参数（batch_size, lr, epoch 等）
    - 随机种子 & 设备(cpu / cuda)
    """

    # ========= 1. 路径相关 =========
    # 项目根目录：就是当前这个 config.py 所在的目录
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 数据根目录：之前你说数据在 data/ 下面
    DATA_DIR = PROJECT_ROOT / "data"

    # 标注文件路径（之后我们会写脚本生成这些 txt）
    # 每一行格式：  图片相对路径 \t latex字符串
    TRAIN_LABELS = DATA_DIR / "train.txt"
    VAL_LABELS = DATA_DIR / "val.txt"
    TEST_LABELS = DATA_DIR / "test.txt"  # 可选，有就用，没有可以先不管

    # 例如之前说过的 “整合的 ICDAR 全量标注文件”
    FULL_LABELS = DATA_DIR / "train_full_icdar.txt"

    # 图片根目录（假设标注文件里写的是类似 "train/img_0001.png" 这种相对路径）
    IMG_BASE_DIR = PROJECT_ROOT

    # 模型和日志目录
    CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "results" / "logs"

    # ========= 2. 图像 & 预处理相关 =========
    # 手写公式图像一般是“宽 > 高”的长条，这里统一缩放到固定高，宽度按比例缩放再 pad 到 MAX_WIDTH
    IMG_HEIGHT = 64  # 统一的高度（像素）
    MAX_WIDTH = 512  # 统一的最大宽度（像素），不足的右侧 pad，超过的会被截断或缩放
    NUM_CHANNELS = 1  # 手写灰度图通常是单通道，如果你读成 RGB 就改成 3

    # ========= 3. 文本编码相关（LaTeX -> 索引） =========
    # 字符表文件（我们后面会写一个脚本从 FULL_LABELS 里自动统计所有 token 并生成）
    VOCAB_PATH = DATA_DIR / "vocab.txt"

    # 特殊符号
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"  # start of sequence
    EOS_TOKEN = "<EOS>"  # end of sequence
    UNK_TOKEN = "<UNK>"  # unknown

    # ========= 4. 模型超参数 =========
    # CNN 特征图的通道数（输出维度）
    CNN_OUT_CHANNELS = 256

    # Transformer / Encoder-Decoder 相关
    D_MODEL = 256  # 特征向量维度
    N_HEAD = 8  # Multi-head Attention 头数
    NUM_ENCODER_LAYERS = 4  # 编码器层数
    NUM_DECODER_LAYERS = 4  # 解码器层数
    DIM_FF = 512  # 前馈网络隐层维度
    DROPOUT = 0.1

    # ========= 5. 训练超参数 =========
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # 学习率调整相关（后面训练脚本里会用到）
    LR_STEP_SIZE = 15  # 每多少个 epoch 衰减一次
    LR_GAMMA = 0.1  # 衰减系数

    # ========= 6. 设备 & 随机种子 =========
    SEED = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed(seed: int = None):
        """
        统一设置随机种子，保证可复现性。
        之后在 train 脚本里会调用：Config.set_seed()
        """
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 让 cudnn 的结果尽可能可复现（会略微牺牲一点速度）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
