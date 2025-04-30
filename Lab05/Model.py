import torch.nn as nn
import torch.nn.functional as F

class Net_Animals(nn.Module):
    def __init__(self, num_classes):
        super(Net_Animals, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)

        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = F.relu(x)
        x = self.conv2(x); x = self.bn2(x); x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv3(x); x = self.bn3(x); x = F.relu(x)
        x = self.conv4(x); x = self.bn4(x); x = F.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.dropout1(x); x = self.fc1(x); x = self.bn5(x); x = F.relu(x)
        x = self.dropout2(x); x = self.fc2(x); x = self.bn6(x); x = F.relu(x)
        x = self.dropout3(x); x = self.fc3(x)
        return F.log_softmax(x, dim=1)