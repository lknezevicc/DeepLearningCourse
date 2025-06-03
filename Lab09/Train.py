from Model import MusicModel
from Prepare_Data import extract_notes_from_folder, prepare_sequences, MusicDataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  notes = extract_notes_from_folder("data")
  inputs, targets = prepare_sequences(notes)
  encoder = LabelEncoder()
  encoder.fit(notes)
  dataset = MusicDataset(inputs, targets, encoder)
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

  model = MusicModel(vocab_size=len(encoder.classes_)).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 10
  for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      output, _ = model(x)
      loss = criterion(output, y)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      _, pred = torch.max(output, 1)
      total += y.size(0)
      correct += (pred == y).sum().item()

    print(f"Epoch {epoch+1}: Loss = {running_loss/len(dataloader):.4f}, Accuracy = {100*correct/total:.2f}%")

  torch.save(model.state_dict(), "music_model.pth")

if __name__ == "__main__":
	main()