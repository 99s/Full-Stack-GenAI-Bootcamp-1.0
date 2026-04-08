# /// script
# dependencies = [
#   "selenium",
#   "beautifulsoup4",
#   "pandas",
#   "webdriver-manager",
# ]
# ///

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_flipkart(url, target_count=100):
    chrome_options = Options()
    # If you get 0 results, comment out the headless line to debug visually
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    reviews_list = []
    
    try:
        driver.get(url)
        
        while len(reviews_list) < target_count:
            # FIX: Corrected method name here
            WebDriverWait(driver, 15).until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'EPCmZ')] | //div[@class='t-ZTKy']"))
            )
            
            # Briefly scroll to bottom to trigger any lazy-loading content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            # Flipkart uses t-ZTKy for the review text container
            containers = soup.find_all('div', {'class': 't-ZTKy'})
            
            for container in containers:
                # Clean up "Read More" text often appended to long reviews
                text = container.get_text(separator=" ", strip=True).replace("READ MORE", "")
                if text and text not in reviews_list:
                    reviews_list.append(text)
                
                if len(reviews_list) >= target_count:
                    break
            
            print(f"Progress: {len(reviews_list)}/{target_count} reviews collected.")

            if len(reviews_list) < target_count:
                try:
                    # Target the 'Next' button specifically
                    next_button = driver.find_element(By.XPATH, "//span[text()='Next']/..")
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(2) 
                except Exception:
                    print("No more pages found.")
                    break

    finally:
        driver.quit()

    df = pd.DataFrame(reviews_list, columns=['review_text'])
    df.to_csv('product_reviews.csv', index=False, encoding='utf-8')
    print(f"✅ Mission Accomplished: {len(df)} reviews saved to product_reviews.csv")

if __name__ == "__main__":
    url = "https://www.flipkart.com/ai-pulse-1-blue-64-gb/product-reviews/itm039eaa37b8fdb?pid=MOBHDDQXW4ZVQFFX&lid=LSTMOBHDDQXW4ZVQFFXZCFXAD&marketplace=FLIPKART"
    scrape_flipkart(url)