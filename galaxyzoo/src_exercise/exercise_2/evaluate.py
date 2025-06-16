import torch
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_regression(model, loader, device="cpu"):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = mean_squared_error(all_targets[:, 2:7], all_preds)
    return mse, all_preds, all_targets