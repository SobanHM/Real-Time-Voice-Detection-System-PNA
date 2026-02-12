import json
import re
from collections import defaultdict

def normalize_text(text):
  # strip + collapse multiple internal spaces
  return re.sub(r'\s+', ' ', text.strip())  

# load original data
with open("mna_list.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Clean and normalize entries
cleaned_data = []
for member in raw_data:
    cleaned_data.append({
        "name": normalize_text(member["name"]),
        "party": normalize_text(member["party"]).upper()  # Optional: convert to uppercase
    })

with open("cleaned_mna_data.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=4)

# count members per party
party_counts = defaultdict(int)
for member in cleaned_data:
    party_counts[member["party"]] += 1

# print summary
print("Party-wise Member Count:")
for party, count in sorted(party_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{party}: {count}")
