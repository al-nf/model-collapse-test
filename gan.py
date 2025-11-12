#!/usr/bin/env python3
import os
import math
import random
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from PIL import Image
import numpy as np

NUM_EPOCHS = 30
MINI_BATCH_SIZE = 32
LEARN_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.999
FLIP_FACTOR = 0.3
VALIDATION_FREQUENCY = 20
NUM_LATENT = 100
IMAGE_SIZE = 64
NUM_VALIDATION_IMAGES = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


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


def save(tensor, out_dir, prefix: Optional[str] = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor = (tensor + 1) / 2
    for i, img in enumerate(tensor):
        if prefix:
            fname = f'gen_{prefix}_{i:04d}.png'
        else:
            fname = f'gen_{i:04d}.png'
        path = out_dir / fname
        utils.save_image(img, str(path))

    return out_dir

def make_mixed_dataloader(real_folder: Path, gen_folder: Optional[Path], gen_ratio: float, batch_size=MINI_BATCH_SIZE, max_real_images=2000):
    if not real_folder.exists():
        return None
    
    real_imgs = [p for p in real_folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    if len(real_imgs) == 0:
        return None
    
    if len(real_imgs) > max_real_images:
        real_imgs = random.sample(real_imgs, max_real_images)
        print(f"Subsampled dataset from full size to {max_real_images} images")
    
    gen_imgs = []
    if gen_folder is not None and gen_folder.exists():
        gen_imgs = [p for p in gen_folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    
    if len(gen_imgs) > 0 and gen_ratio > 0:
        num_real = int(len(real_imgs) * (1 - gen_ratio))
        num_gen = int(len(real_imgs) * gen_ratio)
        
        num_real = max(1, num_real)
        num_gen = min(len(gen_imgs), num_gen)
        
        selected_real = random.sample(real_imgs, min(num_real, len(real_imgs)))
        selected_gen = random.sample(gen_imgs, num_gen)
        
        all_imgs = selected_real + selected_gen
        print(f"Mixed dataset: {len(selected_real)} real + {len(selected_gen)} generated = {len(all_imgs)} total")
    else:
        all_imgs = real_imgs
        print(f"Pure real dataset: {len(all_imgs)} images")
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), 
        transforms.Lambda(lambda t: t * 2 - 1) 
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

    ds = SimpleFolder(all_imgs, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return dl


def train_for_folder(round_num: int, path_real: Path, path_gen_prev: Optional[Path], path_gen_out: Path, gen_ratio: float):
    path_gen_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ROUND {round_num}: Training with {gen_ratio*100:.1f}% generated data")
    print(f"{'='*60}")

    dataloader = make_mixed_dataloader(path_real, path_gen_prev, gen_ratio, batch_size=MINI_BATCH_SIZE)
    if dataloader is None:
        print(f'No images available, skipping round {round_num}')
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

            real_targets = torch.ones_like(d_real, device=device)
            fake_targets = torch.zeros_like(d_fake, device=device)

            num_obs = real_targets.shape[0]
            num_to_flip = int(math.floor(FLIP_FACTOR * num_obs))
            if num_to_flip > 0:
                flip_idx = random.sample(range(num_obs), num_to_flip)
                real_targets[flip_idx] = 0.0

            criterion = nn.BCEWithLogitsLoss()
            lossD = criterion(d_real, real_targets) + criterion(d_fake, fake_targets)

            optimD.zero_grad()
            lossD.backward()
            optimD.step()

            d_fake_forG = netD(netG(z))
            real_targets_forG = torch.ones_like(d_fake_forG, device=device)
            lossG2 = criterion(d_fake_forG, real_targets_forG)
            optimG.zero_grad()
            lossG2.backward()
            optimG.step()

            with torch.no_grad():
                prob_real = torch.sigmoid(d_real)
                prob_fake = torch.sigmoid(d_fake)
                scoreD = ((prob_real.mean() + (1 - prob_fake.mean())) / 2).item()
                scoreG = prob_fake.mean().item()

            if iteration % VALIDATION_FREQUENCY == 0 or iteration == 1:
                with torch.no_grad():
                    gval = netG(z_val)
                    if epoch > (NUM_EPOCHS // 2):
                        try:
                            prefix = f'round{round_num}_iter{iteration:06d}_ep{epoch:03d}'
                            save(gval[:NUM_VALIDATION_IMAGES], path_gen_out, prefix=prefix)
                        except Exception:
                            pass

            if iteration % 10 == 0:
                print(f'Round {round_num} | Epoch {epoch} Iter {iteration} | G_loss {lossG2.item():.4f} D_loss {lossD.item():.4f} | sG {scoreG:.4f} sD {scoreD:.4f}')
                logging.info(f'Round {round_num} | G_loss {lossG2.item():.4f} | D_loss {lossD.item():.4f}')

    print(f'Finished round {round_num}')
    
    with torch.no_grad():
        z_final = torch.randn(NUM_VALIDATION_IMAGES, NUM_LATENT, device=device)
        gval_final = netG(z_final)
        save(gval_final, path_gen_out, prefix=f'round{round_num}_final')
    
    return netG


def main():
    NUM_ROUNDS = 10 
    path_real = Path('./data/training')
    base_gen_path = Path('./data/gen')
    
    if not path_real.exists() or len(list(path_real.glob('*.*'))) == 0:
        print(f"Error: No training images found in {path_real}")
        print("Please place training images in ./data/training/")
        return
    
    print(f"Starting {NUM_ROUNDS} rounds of training with progressive data injection")
    print(f"Real training data: {path_real}")
    
    path_gen_prev = None
    
    for round_num in range(1, NUM_ROUNDS + 1):
        gen_ratio = (round_num - 1) * 0.10
        
        path_gen_out = base_gen_path / f'round_{round_num}'
        
        train_for_folder(
            round_num=round_num,
            path_real=path_real,
            path_gen_prev=path_gen_prev,
            path_gen_out=path_gen_out,
            gen_ratio=gen_ratio
        )
        
        path_gen_prev = path_gen_out
        
        print(f"\nRound {round_num} complete. Generated images saved to: {path_gen_out}")
    
    print(f"\n{'='*60}")
    print("All rounds complete!")
    print(f"Results saved in: {base_gen_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
