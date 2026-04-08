import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_reviews(url, target_count=100):
    # Setup Chrome options for headless mode (runs in background)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    reviews_list = []
    driver.get(url)
    time.sleep(3) # Wait for initial load

    while len(reviews_list) < target_count:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Selectors for Amazon (may need adjustment based on specific layout)
        # Amazon: "span[data-hook='review-body']" | Flipkart: "div.t-ZTKy"
        review_elements = soup.select("span[data-hook='review-body']")
        
        for item in review_elements:
            text = item.get_text(strip=True)
            if text and text not in reviews_list:
                reviews_list.append(text)
            if len(reviews_list) >= target_count:
                break

        print(f"Scraped {len(reviews_list)} reviews...")

        # Find and click the 'Next Page' button
        try:
            # Amazon 'Next' button selector
            next_button = driver.find_element(By.CSS_SELECTOR, "li.a-last a")
            next_button.click()
            time.sleep(2) # Ethical delay to avoid being blocked
        except Exception:
            print("No more pages found or blocked by CAPTCHA.")
            break

    driver.quit()
    
    # Store in DataFrame and export to CSV
    df = pd.DataFrame(reviews_list, columns=['review_text'])
    df.to_csv('product_reviews.csv', index=False, encoding='utf-8')
    print("Data saved to product_reviews.csv")

# Replace with a valid Amazon/Flipkart "See All Reviews" URL
# https://www.amazon.in/product-reviews/B0G26FN2MK/ref=cm_cr_dp_d_show_all_btm?ie=UTF8
# https://www.flipkart.com/ai-pulse-1-blue-64-gb/product-reviews/itm039eaa37b8fdb?pid=MOBHDDQXW4ZVQFFX&lid=LSTMOBHDDQXW4ZVQFFXZCFXAD&marketplace=FLIPKART
product_url = "https://www.flipkart.com/ai-pulse-1-blue-64-gb/product-reviews/itm039eaa37b8fdb?pid=MOBHDDQXW4ZVQFFX&lid=LSTMOBHDDQXW4ZVQFFXZCFXAD&marketplace=FLIPKART"
scrape_reviews(product_url)