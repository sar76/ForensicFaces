# Refactored DcGAN for blazing-fast preprocessing and multi-GPU training
# Following YT Tutorial by InsightsByRish to make DcGAN Face Generation model
# Using this as prep for a much larger application for Forensic application
# Using FlickrFaces DataSet -- Big Thanks to InsightsByRish & Matheus Eduardo

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import random

# Enable mixed-precision and XLA for optimal performance
torch.backends.cudnn.benchmark = True

# Constants
DATA_DIR = '64x64/faces'
IMG_SIZE = 64
BUFFER_SIZE = 10000
BATCH_SIZE = 128
EPOCHS = 60
LATENT_DIM = 100
INPUT_SHAPE = (3, 64, 64)
SAMPLE_DIR = './samples'
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Force both sides to 64
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class SingleFolderImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label for compatibility

dataset = SingleFolderImageDataset(root_dir=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 8*8*512),
            nn.ReLU(True),
            nn.BatchNorm1d(8*8*512),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(8*8*256, 1)
        )
    def forward(self, x):
        return self.net(x)

# Explicit weight initialization for Conv and BatchNorm layers (DCGAN best practice)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss and optimizers
criterion = nn.BCEWithLogitsLoss()
# Set both G and D learning rates to 2e-4 (DCGAN paper default)
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
# Add ReduceLROnPlateau schedulers for both G and D
schedulerG = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, 'min', factor=0.5, patience=5)
schedulerD = optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, 'min', factor=0.5, patience=5)

# Fixed noise for sampling
fixed_noise = torch.randn(16, LATENT_DIM, device=device)

# Label smoothing and instance noise parameters
real_label = 0.9  # Label smoothing for real labels
fake_label = 0.1  # Label smoothing for fake labels
instance_noise_sigma = 0.02  # Small instance noise for D inputs

# Training loop
def train():
    for epoch in range(EPOCHS):
        g_running_loss, d_running_loss = 0.0, 0.0  # For scheduler
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            # Use smoothed labels
            real_labels = torch.full((batch_size, 1), real_label, device=device)
            fake_labels = torch.full((batch_size, 1), fake_label, device=device)

            # Add instance noise to D's inputs
            real_imgs_noisy = real_imgs + instance_noise_sigma * torch.randn_like(real_imgs)

            # Train Discriminator
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_imgs = generator(z)
            fake_imgs_noisy = fake_imgs + instance_noise_sigma * torch.randn_like(fake_imgs)
            d_real = discriminator(real_imgs_noisy)
            d_fake = discriminator(fake_imgs_noisy.detach())
            loss_real = criterion(d_real, real_labels)
            loss_fake = criterion(d_fake, fake_labels)
            d_loss = loss_real + loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_imgs = generator(z)
            fake_imgs_noisy = fake_imgs + instance_noise_sigma * torch.randn_like(fake_imgs)
            d_fake = discriminator(fake_imgs_noisy)
            g_loss = criterion(d_fake, real_labels)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_running_loss += g_loss.item()
            d_running_loss += d_loss.item()

            if i % 200 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {i}/{len(dataloader)}\tLoss_D: {d_loss.item():.4f}\tLoss_G: {g_loss.item():.4f}")

        # Save sample images
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
            utils.save_image((fake+1)/2, f"{SAMPLE_DIR}/image_at_epoch_{epoch+1:04d}.png", nrow=4)

        # Step LR schedulers at end of each epoch
        schedulerG.step(g_running_loss)
        schedulerD.step(d_running_loss)

    # Save final model checkpoints
    torch.save(generator.state_dict(), './generator_final.pth')
    torch.save(discriminator.state_dict(), './discriminator_final.pth')

if __name__ == "__main__":
    train() 