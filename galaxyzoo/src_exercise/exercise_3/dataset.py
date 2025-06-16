import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class GalaxyZooRegressionDataset(Dataset):
    def __init__(self, images_folder, labels_csv, transform=None):
        self.labels = pd.read_csv(labels_csv)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = str(self.labels['GalaxyID'].iloc[idx]) + ".jpg"
        img_path = os.path.join(self.images_folder, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        # Select the 14 regression columns for exercise 3
        label_cols = [
            'Class2.1', 'Class2.2', 'Class6.1', 'Class6.2',
            'Class7.1', 'Class7.2', 'Class7.3',
            'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7'
        ]
        label = self.labels.loc[idx, label_cols].values.astype('float32')
        return image, label