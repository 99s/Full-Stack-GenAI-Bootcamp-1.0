# flipkart_api_scraper.py

from playwright.sync_api import sync_playwright
import json


class FlipkartAPIScraper:

    def __init__(self):
        self.reviews = []

    def handle_response(self, response):
        try:
            url = response.url

            # Look for review API calls
            if "review" in url.lower():
                data = response.json()

                # DEBUG: print structure once
                print("Captured API:", url)

                # Try extracting reviews (structure may vary)
                if isinstance(data, dict):
                    text = json.dumps(data)

                    # crude but effective extraction
                    if "review" in text.lower():
                        self.reviews.append(text[:500])

        except:
            pass

    def scrape(self, url):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()

            page.on("response", self.handle_response)

            page.goto(url)
            page.wait_for_timeout(8000)

            # Scroll to trigger API calls
            for _ in range(5):
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(2000)

            browser.close()

        return self.reviews


if __name__ == "__main__":
    url = "https://www.flipkart.com/ai-pulse-1-blue-64-gb/product-reviews/itm039eaa37b8fdb"

    scraper = FlipkartAPIScraper()
    reviews = scraper.scrape(url)

    print(f"Captured {len(reviews)} API responses")