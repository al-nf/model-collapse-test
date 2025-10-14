import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

base_in = "./data/clips"
base_out = "./data/real"

# Parameters
sr = 22050          # sample rate
n_fft = 1024        # window size
hop_length = 512    
n_mels = 128        # mel bands

for idx in range(12): # modify to nr of speakers
    indir = os.path.join(base_in, str(idx))
    outdir = os.path.join(base_out, str(idx))
    os.makedirs(outdir, exist_ok=True)

    for fname in os.listdir(indir):
        if not fname.endswith(".mp3"):
            continue

        infile = os.path.join(indir, fname)
        outfile = os.path.join(outdir, fname + ".jpg")

        if os.path.exists(outfile):
            print(f"Skipping {fname}, spectrogram already exists.")
            continue

        # load
        y, fs = librosa.load(infile, sr=sr, mono=True)

        S = librosa.feature.melspectrogram(
            y=y,
            sr=fs,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # log
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(8, 4))
        librosa.display.specshow(S_dB, sr=fs, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel spectrogram")
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
