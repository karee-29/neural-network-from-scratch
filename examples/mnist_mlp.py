import argparse
import numpy as np
from nnfs.data.mnist import load_mnist, DataLoader
from nnfs.nn import Tensor, Sequential, Linear, ReLU, Dropout, CrossEntropyLoss, Adam
from nnfs.utils.training import train_one_epoch, evaluate, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--train-limit", type=int, default=None)
parser.add_argument("--test-limit", type=int, default=None)
args = parser.parse_args()

X_train, y_train, X_test, y_test = load_mnist(
    limit_train=args.train_limit, limit_test=args.test_limit
)

model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.15),
    Linear(256, 128),
    ReLU(),
    Dropout(0.10),
    Linear(128, 10),
)
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5, grad_clip=5.0)

history = []
for epoch in range(1, args.epochs+1):
    loader = DataLoader(X_train, y_train, args.batch_size, shuffle=True, seed=42+epoch)
    tr_loss, tr_acc = train_one_epoch(model, loader, loss_fn, optimizer)
    te_loss, te_acc = evaluate(model, X_test, y_test, loss_fn)
    history.append({
        "epoch": epoch, "train_loss": tr_loss, "train_accuracy": tr_acc,
        "test_loss": te_loss, "test_accuracy": te_acc
    })
    print(f"epoch={epoch:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
          f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}")

save_checkpoint(
    "checkpoints/mnist_mlp.npz", model, history,
    {"epochs":args.epochs,"batch_size":args.batch_size,"optimizer":"Adam"}
)
