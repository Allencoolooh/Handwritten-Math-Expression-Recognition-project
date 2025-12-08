from config import Config
from utils.vocab import Vocab
from utils.dataset import create_dataloader

def main():
    vocab = Vocab.from_file(Config.VOCAB_PATH)

    train_loader = create_dataloader(
        labels_path=Config.TRAIN_LABELS,
        vocab=vocab,
        batch_size=2,
        shuffle=True,      # 现在可以真·打乱了
        num_workers=0,
        augment=False,
    )

    for batch in train_loader:
        print("images:", batch["images"].shape)
        print("tgt_input:", batch["tgt_input"].shape)
        print("tgt_output:", batch["tgt_output"].shape)
        print("tgt_lengths:", batch["tgt_lengths"])
        print("labels:", batch["labels"][:2])
        break

if __name__ == "__main__":
    main()
