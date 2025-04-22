import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

log_dir = "runs/emnist_" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = nn.Sequential(
    nn.Flatten(),

    nn.Linear(28*28, 256),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.1),

    nn.Linear(256, 128),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.1),

    nn.Linear(128, 64),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.1),

    nn.Linear(64, 27)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10
print("Starting training...")
for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  total_correct = 0
  total_samples = 0

  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    labels -= 1

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    total_correct += (predicted == labels).sum().item()
    total_samples += labels.size(0)

  epoch_loss = running_loss / len(train_loader)
  epoch_acc = 100 * total_correct / total_samples

  print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
  writer.add_scalar("Loss/train", epoch_loss, epoch)
  writer.add_scalar("Accuracy/train", epoch_acc, epoch)

print("Training complete.")


model.eval()
correct = 0
total = 0

with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    labels -= 1

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test accuracy: {test_accuracy:.2f}%")
writer.add_scalar("Accuracy/test", test_accuracy)

writer.close()