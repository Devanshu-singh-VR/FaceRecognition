import torch
import torch.nn as nn
import torchvision.models as models

class Siamese(nn.Module):
    def __init__(self, in_channels, out_size, dropout):
        super(Siamese, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        #self.model = models.vgg16(pretrained=False)
        #self.model.classifier[6] = nn.Linear(4096, 2148)
        #self.fc_out = nn.Linear(2148, out_size)
        self.model = self.cnn(in_channels)
        self.fc = nn.Linear(256*8*8, 300)
        self.fc_out = nn.Linear(300, out_size)

    def cnn(self, in_channels):
        model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return model

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.model(x).reshape(batch_size, -1)
        x = self.dropout_layer(self.fc(x))
        x = self.fc_out(x)
        #x = self.model(x)
        #x = self.fc_out(x)
        return x

if __name__ == '__main__':
    model = Siamese(3, 100, 0.1)
    x = torch.ones((64, 3, 128, 128))
    print(model(x).shape)