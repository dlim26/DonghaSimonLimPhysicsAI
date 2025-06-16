import torch

def exercise3_train(model, loader, optimizer, loss_fn, output_weights=None, device="cpu"):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y, output_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def exercise3_validate(model, loader, loss_fn, output_weights=None, device="cpu"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y, output_weights)
            total_loss += loss.item()
    return total_loss / len(loader)