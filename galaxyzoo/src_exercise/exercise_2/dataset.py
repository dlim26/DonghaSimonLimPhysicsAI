import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class GalaxyZooRegressionDataset(Dataset):
    def __init__(self, images_folder, labels_csv, transform=None):
        self.labels = pd.read_csv(labels_csv)
        self.images_folder = images_folder
        self.transform = transform
        
        self.label_cols = [
            'Class1.1', 'Class1.2', 'Class2.1', 'Class2.2', 'Class7.1', 'Class7.2', 'Class7.3'
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = str(int(self.labels['GalaxyID'].iloc[idx])) + ".jpg"
        img_path = os.path.join(self.images_folder, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels.loc[idx, self.label_cols].values.astype('float32')
        return image, label