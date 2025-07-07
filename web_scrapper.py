import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ", strip=True)
        return text if text else f"No readable text extracted from {url}."
    except Exception as e:
        print(f"Error scraping website: {e}")
        return f"Failed to extract content from {url}."
