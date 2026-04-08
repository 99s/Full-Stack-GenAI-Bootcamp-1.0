# playwright_scraper.py

from playwright.sync_api import sync_playwright
import pandas as pd
import time


class FlipkartScraperPW:

    def __init__(self):
        self.reviews = []

    def scrape(self, url, max_reviews=100):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()

            page.goto(url)
            time.sleep(5)

            # Close login popup if present
            try:
                page.click("button:has-text('✕')", timeout=3000)
            except:
                pass

            while len(self.reviews) < max_reviews:
                page.wait_for_selector("div.t-ZTKy", timeout=10000)

                elements = page.query_selector_all("div.t-ZTKy")

                for el in elements:
                    text = el.inner_text().replace("READ MORE", "").strip()
                    if text and text not in self.reviews:
                        self.reviews.append(text)

                    if len(self.reviews) >= max_reviews:
                        break

                print(f"Collected: {len(self.reviews)}")

                # Next page
                try:
                    page.click("text=Next")
                    time.sleep(3)
                except:
                    break

            browser.close()

        return self.reviews

    def save(self, filename="reviews.csv"):
        df = pd.DataFrame(self.reviews, columns=["review"])
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} reviews")


if __name__ == "__main__":
    url = "https://www.flipkart.com/ai-pulse-1-blue-64-gb/product-reviews/itm039eaa37b8fdb"

    scraper = FlipkartScraperPW()
    reviews = scraper.scrape(url, max_reviews=100)
    scraper.save("flipkart_reviews.csv")