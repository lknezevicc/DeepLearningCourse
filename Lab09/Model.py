import torch.nn as nn

class MusicModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, num_layers=2):
    super(MusicModel, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, vocab_size)

  def forward(self, x, hidden=None):
    x = self.embedding(x)
    out, hidden = self.lstm(x, hidden)
    out = self.fc(out[:, -1, :])
    return out, hidden