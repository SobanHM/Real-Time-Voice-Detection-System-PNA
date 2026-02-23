import noisereduce as nr
import librosa
import soundfile as sf
from pathlib import Path

def reduce_noise(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(wav_path, reduced_noise, sr)