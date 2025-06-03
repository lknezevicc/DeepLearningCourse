import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Model import resnet

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  
  train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
  val_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transform)

  batch_size = 128
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

  model = resnet(num_classes=10).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

  writer = SummaryWriter('runs/transfer_learning_experiment')

  num_epochs = 30
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      accuracy = 100 * correct / total
      progress_bar.set_postfix({'Accuracy': f"{accuracy:.2f}%"})

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    scheduler.step()

    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)

    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")

  print("Training complete!")

  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validating"):
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  val_accuracy = 100 * correct / total
  print(f"Accuracy on validation dataset: {val_accuracy:.2f}%")
  writer.add_scalar('Validation/Accuracy', val_accuracy)

  writer.close()

if __name__ == "__main__":
  main()