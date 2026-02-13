import librosa
import os
import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(r"D:\RealTimeVoiceDetection_PNA\dataset_&_Model_Developement\dataset.csv")
BASE_DIR = Path(r"D:\RealTimeVoiceDetection_PNA\DataExtraction")

N_MFCC = 13


# load dataset.csv
df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

features = []
labels = []

for index, row in df.iterrows():
    rel_path = row["path"]
    label = row["label"]

    full_path = BASE_DIR / rel_path
    if not str(full_path).endswith(".wav"):
        continue

    try:
        y, sr = librosa.load(full_path)
        if len(y) == 0:
            print(f"=> Skipping empty file: {full_path}")
            continue

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        features.append(mfcc_mean)
        labels.append(label)

    except Exception as e:
        print(f"=> Error processing {full_path}: {e}")

# saving features to .npy
features = np.array(features)
labels = np.array(labels)

np.save("features.npy", features)
np.save("labels.npy", labels)

print("saved features to features.npy and labels to labels.npy")
print(f"Total valid samples: {len(features)}")
