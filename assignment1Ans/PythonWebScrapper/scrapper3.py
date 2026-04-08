# flipkart_scraper.py

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


class FlipkartReviewScraper:

    def __init__(self, headless=False):
        self.options = Options()

        if headless:
            self.options.add_argument("--headless=new")

        self.options.add_argument("--window-size=1280,800")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option("useAutomationExtension", False)

        self.options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.options
        )

        # Bypass webdriver detection
        self.driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """)

        self.wait = WebDriverWait(self.driver, 15)

    def open_page(self, url):
        self.driver.get(url)
        time.sleep(5)  # allow JS load

        self._close_login_popup()

    def _close_login_popup(self):
        try:
            close_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'✕')]"))
            )
            close_btn.click()
            print("Closed login popup")
        except:
            pass

    def _scroll_page(self):
        for _ in range(3):
            self.driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(1)

    def _extract_reviews(self):
        reviews = []

        try:
            self.wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[contains(@class,'t-ZTKy')]")
                )
            )
        except:
            print("Reviews not found on page")
            return reviews

        elements = self.driver.find_elements(
            By.XPATH, "//div[contains(@class,'t-ZTKy')]"
        )

        for el in elements:
            text = el.text.strip().replace("READ MORE", "")
            if text:
                reviews.append(text)

        return reviews

    def _go_to_next_page(self):
        try:
            next_btn = self.driver.find_element(
                By.XPATH, "//span[text()='Next']/.."
            )
            self.driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(3)
            return True
        except:
            return False

    def scrape(self, url, max_reviews=100):
        self.open_page(url)

        all_reviews = set()

        while len(all_reviews) < max_reviews:
            self._scroll_page()

            page_reviews = self._extract_reviews()

            for review in page_reviews:
                all_reviews.add(review)
                if len(all_reviews) >= max_reviews:
                    break

            print(f"Collected: {len(all_reviews)} reviews")

            if len(all_reviews) < max_reviews:
                if not self._go_to_next_page():
                    print("No more pages available")
                    break

        return list(all_reviews)

    def save_to_csv(self, reviews, filename="reviews.csv"):
        df = pd.DataFrame(reviews, columns=["review"])
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"Saved {len(df)} reviews to {filename}")

    def close(self):
        time.sleep(3)
        self.driver.quit()


# =========================
# Run Script
# =========================
if __name__ == "__main__":

    url = "https://www.flipkart.com/ai-pulse-1-blue-64-gb/product-reviews/itm039eaa37b8fdb?pid=MOBHDDQXW4ZVQFFX"

    scraper = FlipkartReviewScraper(headless=False)

    try:
        reviews = scraper.scrape(url, max_reviews=100)
        scraper.save_to_csv(reviews, "flipkart_reviews.csv")
    finally:
        scraper.close()