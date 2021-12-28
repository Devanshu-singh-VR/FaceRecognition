# the accuracy is 0.97 in the 200 test set
import torch
import torch.nn as nn
import torch.optim as optim
from network import Siamese
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import BinaryClassDataset
import matplotlib.pyplot as plt

class BinaryClassification(nn.Module):
    def __init__(self, in_channels, out_size, dropout, device):
        super(BinaryClassification, self).__init__()
        self.siamese = Siamese(in_channels, out_size, dropout).to(device)
        self.fc_out = nn.Linear(out_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, face1, face2):
        output = torch.abs(self.siamese(face1) - self.siamese(face2))
        output = self.sigmoid(self.fc_out(output))
        return output


def save_checkpoint(check_point, path, epoch):
    print('Saving the checkpoint ', epoch)
    torch.save(check_point, path)

def load_checkpoint(checkpoint):
    print('Loading the checkpoint')
    model.load_state_dict((checkpoint['model']))
    optimizer.load_state_dict((checkpoint['optimizer']))

# HyperParameters
load_model = False
epochs = 200
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = 3
out_size = 200
dropout = 0.2
path = 'Binary_.pth.tar'

# Import Data_loader
transforms = transforms.ToTensor()
dataset = BinaryClassDataset('train.csv', transforms)
loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

test_dataset = BinaryClassDataset('test.csv', transforms)
test_loader = DataLoader(test_dataset, batch_size=15, pin_memory=True)

# model
model = BinaryClassification(in_channels, out_size, dropout, device).to(device)

loss_f = nn.BCELoss() # this loss is for sigmoid Binary CrossEntropy loss
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

        score = model(face1_test, face2_test).reshape(-1)
        print(score)
        print(label_test)
        break
    model.train()


    for batch_idx, (face1, face2, label) in enumerate(loader):
        face1 = face1.to(device)
        face2 = face2.to(device)
        label = label.to(device).reshape(-1, 1).type(torch.float32) # khaas the label should be in float format
        score = model(face1, face2)

        optimizer.zero_grad()

        loss = loss_f(score, label)
        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print(f'Loss = {sum(losses)/len(losses)}')


