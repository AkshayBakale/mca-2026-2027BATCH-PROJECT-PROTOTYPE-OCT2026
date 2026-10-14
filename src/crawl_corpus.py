import re
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag

START_URL = "https://en.wikipedia.org/wiki/American_Sign_Language"
WIKI = "https://en.wikipedia.org/wiki/"
MAX_PAGES = 20
DELAY = 1.0                                   # polite crawling
HEADERS = {"User-Agent": "ASL-Project-Crawler/1.0 (student project)"}

queue, visited, sentences = [START_URL], set(), []

while queue and len(visited) < MAX_PAGES:
    url = queue.pop(0)
    if url in visited:
        continue
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
    except requests.RequestException:
        continue
    if resp.status_code != 200:
        continue
    visited.add(url)

    soup = BeautifulSoup(resp.text, "html.parser")
    for p in soup.find_all("p"):                       # extract text
        for sent in re.split(r"[.!?]", p.get_text(" ", strip=True)):
            words = re.findall(r"[a-z]+", sent.lower())
            if len(words) >= 3:
                sentences.append(" ".join(words))

    for a in soup.find_all("a", href=True):            # extract links -> frontier
        link = urldefrag(urljoin(url, a["href"]))[0]
        if link.startswith(WIKI) and ":" not in link[len(WIKI):] \
                and link not in visited and link not in queue:
            queue.append(link)

    print(f"[{len(visited)}/{MAX_PAGES}] crawled {url} | sentences so far: {len(sentences)}")
    time.sleep(DELAY)

os.makedirs("data", exist_ok=True)
with open("data/corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sentences))
print(f"\nSaved {len(sentences)} sentences from {len(visited)} pages to data/corpus.txt")