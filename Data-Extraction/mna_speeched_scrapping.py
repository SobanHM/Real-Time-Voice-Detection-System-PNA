import os
import json
import yt_dlp
from pathlib import Path
import re

DATA_DIR = Path("DataMNAs")
MAX_VIDEOS = 5               # er MNA
CLIP_DURATION = 15          # 15 seconds
CLIP_START = "00:00:15"     # start from 15th second
OFFICIAL_CHANNEL = "National Assembly Of Pakistan"


def sanitize(name):
    return re.sub(r'\W+', '_', name).strip('_')

# used root path (if someone else is using this code) 
def load_mna_list(json_file="cleaned_mna_name_party.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


# downloader function
def download_clips(mna_name):
    folder_name = sanitize(mna_name)
    output_path = DATA_DIR / folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"====Downloading for: {mna_name}")

    ydl_opts = {
        'quiet': True,
        'outtmpl': str(output_path / f"{folder_name}_%(title).80s.%(ext)s"),
        'format': 'bestaudio/best',
        'ignoreerrors': True,
        'noplaylist': True,
        'match_filter': yt_dlp.utils.match_filter_func("duration > 60"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': [
            '-ss', CLIP_START,
            '-t', str(CLIP_DURATION)
        ],
        'max_downloads': MAX_VIDEOS,
        'logger': MyLogger(),
        'progress_hooks': [hook]
    }

    query = f"ytsearch{MAX_VIDEOS}:{mna_name} MNA {OFFICIAL_CHANNEL}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([query])
        except Exception as e:
            print(f"Error for {mna_name}: {e}")

# logging support on youtube
class MyLogger:
    def debug(self, msg): pass
    def warning(self, msg): print(f" {msg}")
    def error(self, msg): print(f"{msg}")

def hook(d):
    if d['status'] == 'finished':
        print(f"== Downloaded: {d['filename']}")

# pipelineeee
if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    mnas = load_mna_list()

    for mna in mnas[:5]:
        download_clips(mna["name"])
