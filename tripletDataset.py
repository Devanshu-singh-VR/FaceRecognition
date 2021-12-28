import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class TripletDataset(Dataset):
    def __init__(self, file_path, transformer):
        super(TripletDataset, self).__init__()
        self.pd = pd.read_csv(file_path)
        self.transformer = transformer
        self.face1 = self.pd['face1']
        self.face2 = self.pd['face2']
        self.face3 = self.pd['face3']

    def __len__(self):
        return len(self.face1)

    def __getitem__(self, index):
        anchor = Image.open(self.face1[index])
        positive = Image.open(self.face2[index])
        negative = Image.open(self.face3[index])

        if self.transformer:
            anchor = self.transformer(anchor)
            positive = self.transformer(positive)
            negative = self.transformer(negative)

        return anchor, positive, negative

if __name__ == '__main__':
    transform = transforms.ToTensor()
    dataset = TripletDataset('train_triplet.csv', transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)

    for a, p, n in data_loader:
        print(a.shape, p.shape, n.shape)