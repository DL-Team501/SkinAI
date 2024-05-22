import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.data.preprocess.clean_ingredients_data import get_formatted_data


class SkinCareProductsDataset(Dataset):
    def __init__(self):
        self.data = get_formatted_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        ingredients_encoded = torch.tensor(sample['ingredients_index_encoded'], dtype=torch.float32)
        skin_types = torch.tensor(sample['skin_types_list'], dtype=torch.float32)

        return ingredients_encoded, skin_types


def create_dataloaders(batch_size=32, train_split=0.8):
    dataset = SkinCareProductsDataset()
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    batch_size = 32
    dataloader = create_dataloaders(batch_size, train=True)

    for batch in dataloader:
        print(batch['ingredients_encoded'].size(), batch['skin_types'].size())
