import torch
from config import Config
from model.encoder import CNNEncoder


def main():
    B = 2
    H, W = Config.IMG_HEIGHT, Config.MAX_WIDTH
    x = torch.randn(B, Config.NUM_CHANNELS, H, W)  # 假装一批随机图

    encoder = CNNEncoder(
        in_channels=Config.NUM_CHANNELS,
        d_model=Config.D_MODEL,
    )
    y = encoder(x)
    print("input shape :", x.shape)
    print("output shape:", y.shape)  # 期望 (B, T, D)


if __name__ == "__main__":
    main()
