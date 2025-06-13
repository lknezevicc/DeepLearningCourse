import torch
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, z_dim=100, img_channels=3, features_g=64):
    super(Generator, self).__init__()
    self.gen = nn.Sequential(
      nn.ConvTranspose2d(z_dim, features_g * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(features_g * 8),
      nn.ReLU(True),

      nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(features_g * 4),
      nn.ReLU(True),

      nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(features_g * 2),
      nn.ReLU(True),

      nn.ConvTranspose2d(features_g * 2, img_channels, 4, 2, 1, bias=False),
      nn.Tanh()
      )

  def forward(self, x):
    return self.gen(x)


class Discriminator(nn.Module):
  def __init__(self, img_channels=3, features_d=64):
    super(Discriminator, self).__init__()
    self.disc = nn.Sequential(
      nn.Conv2d(img_channels, features_d, 4, 2, 1),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(features_d, features_d * 2, 4, 2, 1),
      nn.BatchNorm2d(features_d * 2),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1),
      nn.BatchNorm2d(features_d * 4),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(features_d * 4, 1, 4, 1, 0),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.disc(x)
