import os
import torch


def main():
    data_dir = os.path.dirname(__file__)
    train_path = os.path.join(data_dir, "train1989.pt")
    test_path = os.path.join(data_dir, "test1989.pt")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Missing train1989.pt or test1989.pt")

    # load datasets using weights_only=False to avoid PyTorch 2.6 unpickling error
    Xtr, Ytr = torch.load(train_path, weights_only=False)
    Xte, Yte = torch.load(test_path, weights_only=False)

    print(f"Loaded train set: {Xtr.shape}, {Ytr.shape}")
    print(f"Loaded test set:  {Xte.shape}, {Yte.shape}")


if __name__ == "__main__":
    main()
