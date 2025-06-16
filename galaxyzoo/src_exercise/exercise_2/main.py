from dataset import GalaxyZooRegressionDataset
from model import GalaxyZooRegressor
from train import exercise2_train, exercise2_validate
from evaluate import evaluate_regression
from transforms import transform
from losses import regression_loss

import torch
from torch.utils.data import DataLoader, random_split

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GalaxyZooRegressionDataset(
        images_folder="data/images/transformed_images/",
        labels_csv="data/exercise_2/labels.csv",
        transform=transform()
    )
    val_size = int(0.25 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = GalaxyZooRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_losses, val_losses = [], []
    for epoch in range(30):
        train_loss = exercise2_train(model, train_loader, optimizer, regression_loss, device=device)
        val_loss = exercise2_validate(model, val_loader, regression_loss, device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    mse, all_preds, all_targets = evaluate_regression(model, val_loader, device=device)
    print(f"Final Validation MSE: {mse:.5f}")

if __name__ == "__main__":
    main()