#!/usr/bin/env python3
import os
import math
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from PIL import Image
import numpy as np

NUM_EPOCHS = 160
MINI_BATCH_SIZE = 3
LEARN_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.999
FLIP_FACTOR = 0.3
VALIDATION_FREQUENCY = 20
NUM_LATENT = 100
IMAGE_SIZE = 64
NUM_VALIDATION_IMAGES = 40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        if classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    except Exception:
        pass


class ProjectAndReshape(nn.Module):
    """Mimic the projectAndReshapeLayer: fully connected projection then reshape."""

    def __init__(self, output_size: Tuple[int, int, int], num_channels: int):
        super().__init__()
        self.output_size = output_size
        fc_out = output_size[0] * output_size[1] * output_size[2]
        self.fc = nn.Linear(num_channels, fc_out)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        x = x.view(x.size(0), self.output_size[2], self.output_size[0], self.output_size[1])
        return x


class Generator(nn.Module):
    def __init__(self, num_latent=100, ngf=64, out_channels=3):
        super().__init__()
        self.project = ProjectAndReshape((8, 8, 512), num_latent)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 4 * ngf, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * ngf, ngf, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        if z.dim() == 4:
            z = z.view(z.size(0), z.size(1))
        x = self.project(z)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, in_channels=3, slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, ndf, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(ndf, 2 * ndf, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(2 * ndf),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(4 * ndf),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(4 * ndf, 8 * ndf, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8 * ndf),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(8 * ndf, 1, kernel_size=4, stride=1)
        )

    def forward(self, x):
        return self.net(x)


def sigmoid(x):
    return torch.sigmoid(x)


def gan_loss(prob_real, prob_generated):
    eps = 1e-8
    loss_discriminator = -torch.mean(torch.log(prob_real + eps)) - torch.mean(torch.log(1 - prob_generated + eps))
    loss_generator = -torch.mean(torch.log(prob_generated + eps))
    return loss_generator, loss_discriminator


def make_dataloader(folder: Path, batch_size=3, max_images=150):
    if not folder.exists():
        return None
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    if len(imgs) == 0:
        return None
    if len(imgs) > max_images:
        imgs = random.sample(imgs, max_images)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # [0,1]
        transforms.Lambda(lambda t: t * 2 - 1)  # [-1,1]
    ])

    class SimpleFolder(torch.utils.data.Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            p = self.paths[idx]
            img = Image.open(p).convert('RGB')
            return self.transform(img)

    ds = SimpleFolder(imgs, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return dl



def save_image_grid(tensor, out_path):
    grid = utils.make_grid((tensor + 1) / 2, nrow=8, padding=2)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(ndarr).save(out_path)


def save_generated_batch_as_files(tensor, out_dir, iteration, base_seed=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor = (tensor + 1) / 2
    for i, img in enumerate(tensor):
        fname = f'gen_{i:04d}.png'
        path = out_dir / fname
        utils.save_image(img, str(path))

    return out_dir

def train_for_folder():
    path_real = Path('./data/training')
    path_gen = Path('./data/gen')
    path_gen.mkdir(parents=True, exist_ok=True)

    dataloader = make_dataloader(path_real, batch_size=MINI_BATCH_SIZE, max_images=150)
    if dataloader is None:
        print(f'No images in {path_real}, skipping')
        return

    netG = Generator(num_latent=NUM_LATENT).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optimG = torch.optim.Adam(netG.parameters(), lr=LEARN_RATE, betas=(BETA1, BETA2))
    optimD = torch.optim.Adam(netD.parameters(), lr=LEARN_RATE, betas=(BETA1, BETA2))

    z_val = torch.randn(NUM_VALIDATION_IMAGES, NUM_LATENT, device=device)

    iteration = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        for real_batch in dataloader:
            iteration += 1
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            z = torch.randn(bs, NUM_LATENT, device=device)

            with torch.no_grad():
                pass

            d_real = netD(real_batch)
            fake = netG(z)
            d_fake = netD(fake.detach())

            prob_real = sigmoid(d_real)
            prob_fake = sigmoid(d_fake)

            num_obs = prob_real.shape[0]
            num_to_flip = int(math.floor(FLIP_FACTOR * num_obs))
            if num_to_flip > 0:
                flip_idx = random.sample(range(num_obs), num_to_flip)
                prob_real[flip_idx] = 1 - prob_real[flip_idx]

            lossG, lossD = gan_loss(prob_real, prob_fake)

            optimD.zero_grad()
            lossD.backward(retain_graph=True)
            optimD.step()

            d_fake_forG = netD(netG(z))
            prob_fake_forG = sigmoid(d_fake_forG)
            lossG2, _ = gan_loss(prob_real, prob_fake_forG)
            optimG.zero_grad()
            lossG2.backward()
            optimG.step()

            scoreD = ((prob_real.mean() + (1 - prob_fake.mean())) / 2).item()
            scoreG = prob_fake.mean().item()

            if iteration % VALIDATION_FREQUENCY == 0 or iteration == 1:
                with torch.no_grad():
                    gval = netG(z_val)
                    if epoch > (NUM_EPOCHS // 8):
                        try:
                            save_generated_batch_as_files(gval[:NUM_VALIDATION_IMAGES], path_gen, iteration)
                        except Exception:
                            pass

            if iteration % 10 == 0:
                print(f'Training Epoch {epoch} Iter {iteration} | G_loss {lossG2.item():.4f} D_loss {lossD.item():.4f} | sG {scoreG:.4f} sD {scoreD:.4f}')

    print(f'Finished training on {path_real}')


def main():
    train_for_folder()


if __name__ == '__main__':
    main()
