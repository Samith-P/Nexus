from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

BASE_URL = "https://www.scimagojr.com/journalrank.php"

all_journals = []
seen_titles = set()

for page in range(1, 24):
    print(f"Scraping page {page}...")

    url = f"{BASE_URL}?category=1702&area=1700&page={page}&total_size=441"
    driver.get(url)

    time.sleep(5)  # allow JS & protection to settle

    rows = driver.find_elements(By.CSS_SELECTOR, "#journalranking tr")[1:]

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) < 13:
            continue

        title_elem = cols[1].find_element(By.TAG_NAME, "a")
        title = title_elem.text.strip()

        if title in seen_titles:
            continue
        seen_titles.add(title)

        journal = {
            "title": title,
            "detail_url": title_elem.get_attribute("href"),
            "type": cols[2].text.strip(),
            "sjr_raw": cols[3].text.strip(),
            "h_index": cols[4].text.strip(),
            "total_docs_2024": cols[5].text.strip(),
            "total_docs_3y": cols[6].text.strip(),
            "total_refs_2024": cols[7].text.strip(),
            "total_citations_3y": cols[8].text.strip(),
            "citable_docs_3y": cols[9].text.strip(),
            "citations_per_doc_2y": cols[10].text.strip(),
            "refs_per_doc_2024": cols[11].text.strip(),
            "female_percent_2024": cols[12].text.strip()
        }

        all_journals.append(journal)

    time.sleep(2)

driver.quit()

print("\nTotal journals scraped:", len(all_journals))

with open("journals_scimago_raw.json", "w", encoding="utf-8") as f:
    json.dump(all_journals, f, indent=2)

print("Saved journals_scimago_raw.json")
