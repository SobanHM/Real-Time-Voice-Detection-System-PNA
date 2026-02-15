import librosa
import numpy as np

def extract_features(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)  # load audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # extract MFCC
    mfcc_mean = np.mean(mfcc.T, axis=0)  # temporal mean pooling
    return mfcc_mean.reshape(1, -1)  # reshape for model
