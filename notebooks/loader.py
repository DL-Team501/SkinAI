from torch.utils.data import DataLoader, Dataset
import torch

class IngredientsLoader(Dataset):
  def __init__(self, df):
    self.data = df[['ingredients_index_encoded', 'skin_types_list']]
    self.labels = df['skin_types_list']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return torch.tensor(self.data.iloc[idx]['ingredients_index_encoded'], dtype=torch.float) , torch.tensor(self.labels.iloc[idx], dtype=torch.float)

    

# train_loader = DataLoader(IngredientsLoader(df), batch_size=64, shuffle=True)
# for inputs, labels in train_loader:
#   print(inputs, labels)
#   break