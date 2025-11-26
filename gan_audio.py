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
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
from PIL import Image
import numpy as np
import librosa
import soundfile as sf

NUM_EPOCHS = 30
MINI_BATCH_SIZE = 16
LEARN_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.999
FLIP_FACTOR = 0.0
VALIDATION_FREQUENCY = 20
NUM_DOMAINS = 3  
N_MELS = 80 
SEGMENT_LENGTH = 128
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
NUM_VALIDATION_IMAGES = 15
LAMBDA_CLS = 1.0
LAMBDA_REC = 10.0
LAMBDA_GP = 10.0

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


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(dim_out, affine=True)
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, conv_dim=128, c_dim=3, repeat_num=4, n_mels=80):
        super().__init__()
        
        layers = []
        layers.append(nn.Conv1d(n_mels + c_dim, conv_dim, kernel_size=5, stride=1, padding=2, bias=False))
        layers.append(nn.InstanceNorm1d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=5, stride=2, padding=2, bias=False))
            layers.append(nn.InstanceNorm1d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        
        for i in range(2):
            layers.append(nn.ConvTranspose1d(curr_dim, curr_dim // 2, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False))
            layers.append(nn.InstanceNorm1d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        
        layers.append(nn.Conv1d(curr_dim, n_mels, kernel_size=5, stride=1, padding=2, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1)
        c = c.repeat(1, 1, x.size(2)) 
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, conv_dim=128, c_dim=3, repeat_num=4, n_mels=80):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(n_mels, conv_dim, kernel_size=5, stride=2, padding=2))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv1d(curr_dim, curr_dim * 2, kernel_size=5, stride=2, padding=2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv_src = nn.Conv1d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cls = nn.Conv1d(curr_dim, c_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_src(h)
        out_cls = self.conv_cls(h).mean(dim=2)  
        return out_src, out_cls


def gradient_penalty(discriminator, real_images, fake_images, device):
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    d_interpolates, _ = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def classification_loss(logit, target):
    return F.cross_entropy(logit, target)


def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim, device=labels.device)
    out[np.arange(batch_size), labels.long().cpu()] = 1
    return out


def create_labels(c_org, c_dim, device, selected_attrs=None):
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def save(tensor, out_dir, prefix: Optional[str] = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, mel in enumerate(tensor):
        if prefix:
            fname_img = f'gen_{prefix}_{i:04d}_mel.png'
            fname_wav = f'gen_{prefix}_{i:04d}.wav'
        else:
            fname_img = f'gen_{i:04d}_mel.png'
            fname_wav = f'gen_{i:04d}.wav'
        
        mel_np = mel.cpu().numpy()
        mel_norm = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8)
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_norm, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(str(out_dir / fname_img))
        plt.close()
        
        try:
            mel_db = mel_np * (mel_np.std() + 1e-5) + mel_np.mean()  
            mel_power = librosa.db_to_power(mel_db)
            audio = librosa.feature.inverse.mel_to_audio(
                mel_power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            sf.write(str(out_dir / fname_wav), audio, SAMPLE_RATE)
        except Exception as e:
            print(f"Warning: Could not save audio {i}: {e}")

    return out_dir

class AudioMelDataset(Dataset):
    def __init__(self, audio_paths, num_domains, sr=SAMPLE_RATE, n_fft=N_FFT, 
                 hop_length=HOP_LENGTH, n_mels=N_MELS, segment_length=SEGMENT_LENGTH):
        self.audio_paths = audio_paths
        self.num_domains = num_domains
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.segment_length = segment_length

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sr // 2
        )
        
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
        
        mel_tensor = torch.FloatTensor(log_mel)
        
        if mel_tensor.shape[1] >= self.segment_length:
            start = random.randint(0, mel_tensor.shape[1] - self.segment_length)
            mel_tensor = mel_tensor[:, start:start + self.segment_length]
        else:
            pad_length = self.segment_length - mel_tensor.shape[1]
            mel_tensor = F.pad(mel_tensor, (0, pad_length))
        
        label = torch.randint(0, self.num_domains, (1,)).item()
        
        return mel_tensor, label


def make_mixed_dataloader(real_folder: Path, gen_folder: Optional[Path], gen_ratio: float, batch_size=MINI_BATCH_SIZE, max_real_images=2000):
    if not real_folder.exists():
        return None
    
    real_audio = [p for p in real_folder.iterdir() if p.suffix.lower() in ('.wav', '.flac', '.mp3', '.ogg')]
    if len(real_audio) == 0:
        return None
    
    if len(real_audio) > max_real_images:
        real_audio = random.sample(real_audio, max_real_images)
        print(f"Subsampled dataset from full size to {max_real_images} audio files")
    
    gen_audio = []
    if gen_folder is not None and gen_folder.exists():
        gen_audio = [p for p in gen_folder.iterdir() if p.suffix.lower() in ('.wav', '.flac', '.mp3', '.ogg')]
    
    if len(gen_audio) > 0 and gen_ratio > 0:
        num_real = int(len(real_audio) * (1 - gen_ratio))
        num_gen = int(len(real_audio) * gen_ratio)
        
        num_real = max(1, num_real)
        num_gen = min(len(gen_audio), num_gen)
        
        selected_real = random.sample(real_audio, min(num_real, len(real_audio)))
        selected_gen = random.sample(gen_audio, num_gen)
        
        all_audio = selected_real + selected_gen
        print(f"Mixed dataset: {len(selected_real)} real + {len(selected_gen)} generated = {len(all_audio)} total")
    else:
        all_audio = real_audio
        print(f"Pure real dataset: {len(all_audio)} audio files")
    
    ds = AudioMelDataset(all_audio, NUM_DOMAINS)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
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

    netG = Generator(conv_dim=128, c_dim=NUM_DOMAINS, repeat_num=4, n_mels=N_MELS).to(device)
    netD = Discriminator(conv_dim=128, c_dim=NUM_DOMAINS, repeat_num=4, n_mels=N_MELS).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optimG = torch.optim.Adam(netG.parameters(), lr=LEARN_RATE, betas=(BETA1, BETA2))
    optimD = torch.optim.Adam(netD.parameters(), lr=LEARN_RATE, betas=(BETA1, BETA2))

    fixed_x = []
    fixed_c = []

    iteration = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        for real_x, real_label in dataloader:
            iteration += 1
            real_x = real_x.to(device)
            real_label = real_label.to(device)
            bs = real_x.size(0)

            rand_idx = torch.randperm(real_label.size(0))
            target_label = real_label[rand_idx]
            
            real_c = label2onehot(real_label, NUM_DOMAINS).to(device)
            target_c = label2onehot(target_label, NUM_DOMAINS).to(device)

            out_src, out_cls = netD(real_x)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = classification_loss(out_cls, real_label)

            fake_x = netG(real_x, target_c)
            out_src, out_cls = netD(fake_x.detach())
            d_loss_fake = torch.mean(out_src)

            d_loss_gp = gradient_penalty(netD, real_x, fake_x, device)
            d_loss = d_loss_real + d_loss_fake + LAMBDA_CLS * d_loss_cls + LAMBDA_GP * d_loss_gp

            optimD.zero_grad()
            d_loss.backward()
            optimD.step()

            if iteration % 5 == 0:
                fake_x = netG(real_x, target_c)
                out_src, out_cls = netD(fake_x)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = classification_loss(out_cls, target_label)

                reconstructed_x = netG(fake_x, real_c)
                g_loss_rec = torch.mean(torch.abs(real_x - reconstructed_x))

                g_loss = g_loss_fake + LAMBDA_REC * g_loss_rec + LAMBDA_CLS * g_loss_cls

                optimG.zero_grad()
                g_loss.backward()
                optimG.step()
            else:
                g_loss = torch.tensor(0.0)

            if iteration % VALIDATION_FREQUENCY == 0 or iteration == 1:
                if len(fixed_x) < NUM_VALIDATION_IMAGES and real_x.size(0) > 0:
                    fixed_x.append(real_x[0:1].cpu())
                    fixed_c.append(real_c[0:1].cpu())
                
                if len(fixed_x) >= min(NUM_VALIDATION_IMAGES, 5) and epoch > (NUM_EPOCHS // 4):
                    with torch.no_grad():
                        try:
                            x_concat = []
                            for i in range(min(len(fixed_x), 5)):
                                x_fixed = fixed_x[i].to(device)
                                for j in range(NUM_DOMAINS):
                                    c_trg = label2onehot(torch.tensor([j]), NUM_DOMAINS).to(device)
                                    x_fake = netG(x_fixed, c_trg)
                                    x_concat.append(x_fake)
                            
                            x_concat = torch.cat(x_concat, dim=0)
                            prefix = f'round{round_num}_iter{iteration:06d}_ep{epoch:03d}'
                            save(x_concat, path_gen_out, prefix=prefix)
                        except Exception as e:
                            print(f"Error saving validation images: {e}")

            if iteration % 10 == 0:
                print(f'Round {round_num} | Epoch {epoch} Iter {iteration} | G_loss {g_loss.item():.4f} D_loss {d_loss.item():.4f} | D_real {d_loss_real.item():.4f} D_fake {d_loss_fake.item():.4f}')
                logging.info(f'Round {round_num} | G_loss {g_loss.item():.4f} | D_loss {d_loss.item():.4f}')

    print(f'Finished round {round_num}')
    
    with torch.no_grad():
        if len(fixed_x) > 0:
            x_concat = []
            for i in range(min(len(fixed_x), NUM_VALIDATION_IMAGES)):
                x_fixed = fixed_x[i].to(device)
                for j in range(NUM_DOMAINS):
                    c_trg = label2onehot(torch.tensor([j]), NUM_DOMAINS).to(device)
                    x_fake = netG(x_fixed, c_trg)
                    x_concat.append(x_fake)
            
            if len(x_concat) > 0:
                x_concat = torch.cat(x_concat, dim=0)
                save(x_concat, path_gen_out, prefix=f'round{round_num}_final')
    
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
