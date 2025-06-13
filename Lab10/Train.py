import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
img_channels = 3
batch_size = 128
feature_g = 64
feature_d = 64
lr = 2e-4
num_epochs = 10

transform = transforms.Compose([
  transforms.Resize(32),
  transforms.ToTensor(),
  transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataloader = DataLoader(
  datasets.CIFAR10(root="data", train=True, download=True, transform=transform),
  batch_size=batch_size,
  shuffle=True
)

gen = Generator(z_dim, img_channels, feature_g).to(device)
disc = Discriminator(img_channels, feature_d).to(device)

opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(num_epochs):
  loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
  for batch_idx, (real, _) in enumerate(loop):
    real = real.to(device)
    noise = torch.randn(real.size(0), z_dim, 1, 1).to(device)
    fake = gen(noise)

    disc_real = disc(real).reshape(-1)
    loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
    disc_fake = disc(fake.detach()).reshape(-1)
    loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = (loss_disc_real + loss_disc_fake) / 2

    disc.zero_grad()
    loss_disc.backward()
    opt_disc.step()

    output = disc(fake).reshape(-1)
    loss_gen = criterion(output, torch.ones_like(output))

    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    loop.set_postfix({
      'Loss D': f"{loss_disc.item():.4f}",
      'Loss G': f"{loss_gen.item():.4f}"
    })

def show_generated_images(generator, num_images=16):
  generator.eval()
  with torch.no_grad():
    noise = torch.randn(num_images, z_dim, 1, 1).to(device)
    fake = generator(noise).detach().cpu()
    fake = (fake + 1) / 2
    grid = utils.make_grid(fake, nrow=4)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.show()
  generator.train()

show_generated_images(gen)