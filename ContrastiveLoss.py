import torch
import torch.nn as nn
import torch.optim as optim
from network import Siamese
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataset import BinaryClassDataset

class ConLoss(nn.Module):
    def __init__(self, device, margin=2):
        super(ConLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, output1, output2, label):
        diff = F.pairwise_distance(output1, output2)
        loss = (
                (label * torch.square(diff))
                + ((1 - label) * torch.square(torch.clamp(self.margin - diff, min=0)))
        ).to(self.device)

        return torch.mean(loss)

    def distance(self, output1, output2):
        diff = F.pairwise_distance(output1, output2)
        return diff


def save_checkpoint(check_point, path, epoch):
    print('Saving the checkpoint ', epoch)
    torch.save(check_point, path)

def load_checkpoint(checkpoint):
    print('Loading the checkpoint')
    model.load_state_dict((checkpoint['model']))
    optimizer.load_state_dict((checkpoint['optimizer']))

# HyperParameters
load_model = True
epochs = 200
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 3
out_size = 200
dropout = 0.1
margin = 1
path = 'Contrastive_.pth.tar'

# Import Data_loader
transforms = transforms.ToTensor()
dataset = BinaryClassDataset('train.csv', transforms)
loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

test_dataset = BinaryClassDataset('test.csv', transforms)
test_loader = DataLoader(test_dataset, batch_size=15)

# model
model = Siamese(in_channels, out_size, dropout).to(device)

loss_f = ConLoss(device, margin).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load(path))

for epoch in range(epochs):
    print(f'Epoch[{epoch} / {epochs}]')
    losses = []
    check_points = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(check_points, path, epoch)

    model.eval()
    for idx, (face1_test, face2_test, label_test) in enumerate(test_loader):
        face1_test = face1_test.to(device)
        face2_test = face2_test.to(device)
        label_test = label_test.to(device)

        score = loss_f.distance(
            model(face1_test), model(face2_test)
        )
        print(score)
        print(label_test)
        break
    model.train()


    for batch_idx, (face1, face2, label) in enumerate(loader):
        face1 = face1.to(device)
        face2 = face2.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        loss = loss_f(
            model(face1), model(face2), label
        )
        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print(f'Loss = {sum(losses)/len(losses)}')


