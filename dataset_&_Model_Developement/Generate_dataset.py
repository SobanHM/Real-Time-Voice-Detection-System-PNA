import csv
from pathlib import Path

DATA_DIR = Path("DataMNAs")
CSV_FILE = "dataset.csv"

entries = []

for speaker_dir in DATA_DIR.iterdir():
    if speaker_dir.is_dir():
        speaker = speaker_dir.name
        for wav_file in speaker_dir.glob("*.wav"):
            entries.append([str(wav_file), speaker])

# Save to CSV
with open(CSV_FILE, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])
    writer.writerows(entries)

print(f"====== Dataset CSV saved: {CSV_FILE} ({len(entries)} samples)")
