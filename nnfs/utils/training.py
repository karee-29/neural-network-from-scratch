import json
import numpy as np
from pathlib import Path
from ..nn.tensor import Tensor

def accuracy(logits, y):
    return float((np.argmax(logits.data, axis=1) == y).mean())

def train_one_epoch(model, loader, loss_fn, optimizer, scheduler=None):
    model.train()
    losses, accs = [], []
    for Xb, yb in loader:
        optimizer.zero_grad()
        logits = model(Tensor(Xb))
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        accs.append(accuracy(logits, yb))
    return float(np.mean(losses)), float(np.mean(accs))

def evaluate(model, X, y, loss_fn, batch_size=512):
    model.eval()
    losses, accs = [], []
    for start in range(0, len(X), batch_size):
        logits = model(Tensor(X[start:start+batch_size]))
        loss = loss_fn(logits, y[start:start+batch_size])
        losses.append(loss.item())
        accs.append(accuracy(logits, y[start:start+batch_size]))
    return float(np.mean(losses)), float(np.mean(accs))

def save_checkpoint(path, model, history, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **model.state_dict())
    Path(str(path)+".json").write_text(
        json.dumps({"history": history, "config": config}, indent=2)
    )
