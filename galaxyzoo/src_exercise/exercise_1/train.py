import torch

def exercise1_train(model, loader, optimizer, loss_fn, device="cpu"):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        y_class = y.argmax(dim=1)
        pred = model(X)
        loss = loss_fn(pred, y_class)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(dim=1) == y_class).sum().item()
        total += y.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy
