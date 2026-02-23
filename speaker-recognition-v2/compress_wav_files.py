import os
import subprocess
from pathlib import Path

# path: top-level directory 
BASE_DIR = Path("../wav-files")

for wav_file in BASE_DIR.rglob("*.wav"):
    print(f"Compressing: {wav_file}")
    
    compressed_path = wav_file.with_name(wav_file.stem + "_compressed.wav")
    
    # ffmpeg command: convert to mono (-ac 1) and 16kHz (-ar 16000)
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i", str(wav_file),
        "-ac", "1",
        "-ar", "16000",
        str(compressed_path)
    ]

    try:
        subprocess.run(cmd, check=True)
        # optional: replace original with compressed version
        os.replace(compressed_path, wav_file)
    except subprocess.CalledProcessError as e:
        print(f"Error compressing {wav_file}: {e}")
