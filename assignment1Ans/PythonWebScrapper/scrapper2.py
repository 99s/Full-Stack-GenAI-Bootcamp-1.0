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
    # 1. DISABLED HEADLESS to let you see and solve CAPTCHAs if they appear
    # chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--window-size=1280,800")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    reviews_list = []
    
    try:
        driver.get(url)
        
        while len(reviews_list) < target_count:
            try:
                # 2. Longer wait time and checking for the review container
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "t-ZTKy"))
                )
            except Exception:
                print("Timeout: Elements not found. Saving 'error_debug.png' to see what happened.")
                driver.save_screenshot("error_debug.png")
                break
            
            # Scroll to load elements
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            containers = soup.find_all('div', {'class': 't-ZTKy'})
            
            for container in containers:
                text = container.get_text(separator=" ", strip=True).replace("READ MORE", "")
                if text and text not in reviews_list:
                    reviews_list.append(text)
                if len(reviews_list) >= target_count:
                    break
            
            print(f"Captured: {len(reviews_list)} reviews")

            if len(reviews_list) < target_count:
                try:
                    # Click Next using JavaScript to avoid 'element click intercepted'
                    next_button = driver.find_element(By.XPATH, "//span[text()='Next']/..")
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(3) 
                except:
                    print("No more pages or button hidden.")
                    break

    finally:
        # Give it a second so you can see the final state before it closes
        time.sleep(5) 
        driver.quit()

    df = pd.DataFrame(reviews_list, columns=['review_text'])
    df.to_csv('product_reviews.csv', index=False, encoding='utf-8')
    print(f"Final Count: {len(df)}")

if __name__ == "__main__":
    url = "https://www.flipkart.com/ai-pulse-1-blue-64-gb/product-reviews/itm039eaa37b8fdb?pid=MOBHDDQXW4ZVQFFX&lid=LSTMOBHDDQXW4ZVQFFXZCFXAD&marketplace=FLIPKART"
    scrape_flipkart(url)