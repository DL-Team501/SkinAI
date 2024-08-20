from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.data.preprocess.clean_ingredients_data import get_formatted_data, SkinCareData


class SkinCareProductsDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        tokenized_ingredients = torch.tensor(sample['tokenized_ingredients'], dtype=torch.long)
        skin_types = torch.tensor(sample['one_hot_labels'], dtype=torch.float32)

        return tokenized_ingredients, skin_types


def create_dataloaders(batch_size=32, train_split=0.8) -> Tuple[DataLoader, DataLoader, SkinCareData]:
    skin_care_data = get_formatted_data()
    dataset = SkinCareProductsDataset(skin_care_data.data)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, skin_care_data
