import subprocess
from pathlib import Path

def convert_mp3_to_wav(mp3_path, output_dir):
    output_path = Path(output_dir) / Path(mp3_path).with_suffix(".wav").name
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", "16000", "-ac", "1",
        str(output_path)
    ], check=True)
    return output_path