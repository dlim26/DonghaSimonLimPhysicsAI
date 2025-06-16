from dataset import GalaxyZooExercise1Dataset
from model import Exercise1CNN
from train import exercise1_train
from evaluate import evaluate_classification
from transforms import transform

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GalaxyZooExercise1Dataset(
        images_folder="data/exercise_1/images",
        labels_csv="data/exercise_1/labels_no_class13.csv",
        transform=transform()
    )
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Exercise1CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(30):
        train_loss, train_acc = exercise1_train(model, train_loader, optimizer, loss_fn, device=device)
        print(f"[Epoch {epoch+1}] Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        val_acc, val_cm, val_report = evaluate_classification(model, val_loader, device=device)
        print(f"[Epoch {epoch+1}] Val Acc={val_acc:.4f}")

    val_acc, val_cm, val_report = evaluate_classification(model, val_loader, device=device)
    print("Final Validation Accuracy:", val_acc)
    print("Confusion Matrix:\n", val_cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm)
    disp.plot(cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.show()
    print("Classification Report:\n", val_report)

if __name__ == "__main__":
    main()