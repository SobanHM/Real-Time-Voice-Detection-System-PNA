# scrape_mnas_json.py
import requests
from bs4 import BeautifulSoup
import json

def scrape_mna_name_party():
    url = "https://na.gov.pk/en/all-members.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    table = soup.find("table")
    if not table:
        print("Table not found!")
        return []

    rows = table.find_all("tr")[1:]  # Skip header row
    mna_list = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            name = cols[1].text.strip()
            party = cols[2].text.strip()
            if name and party:
                mna_list.append({"name": name, "party": party})

    return mna_list

if __name__ == "__main__":
    mnas = scrape_mna_name_party()

    if mnas:
        with open("mna_list.json", "w", encoding="utf-8") as f:
            json.dump(mnas, f, indent=2, ensure_ascii=False)

        print(f"Extracted {len(mnas)} MNAs with name and party.")
    else:
        print("No data found.")
