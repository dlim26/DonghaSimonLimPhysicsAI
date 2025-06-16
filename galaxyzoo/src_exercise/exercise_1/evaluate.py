import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_classification(model, loader, device="cpu"):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_class = y.argmax(dim=1)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_class.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=["Class1.1", "Class1.2"])
    return acc, cm, report
