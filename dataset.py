import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class BinaryClassDataset(Dataset):
    def __init__(self, file_path, transformer):
        super(BinaryClassDataset, self).__init__()
        self.pd = pd.read_csv(file_path)
        self.transformer = transformer
        self.face1 = self.pd['face1']
        self.face2 = self.pd['face2']
        self.label = self.pd['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        positive = Image.open(self.face1[index])
        negative = Image.open(self.face2[index])
        label = self.label[index]

        if self.transformer:
            positive = self.transformer(positive)
            negative = self.transformer(negative)

        return positive, negative, torch.tensor(int(label))

if __name__ == '__main__':
    transform = transforms.ToTensor()
    dataset = BinaryClassDataset('train.csv', transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)

    for p, n, label in data_loader:
        print(p.shape, n.shape, label.shape)