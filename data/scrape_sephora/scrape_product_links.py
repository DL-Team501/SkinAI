import json

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Set up Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")

# Set up the WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
wait = WebDriverWait(driver, 10)

# URL of the page to scrape
url = "https://www.sephora.com/shop/skincare"

# Open the page
driver.get(url)


# Function to close modals
def close_modals():
    modal_close_selectors = [
        (By.XPATH, '/html/body/div[5]/div[2]/div/div[2]/div/div/button'),
        (By.XPATH, '/html/body/div[5]/div/div/div[2]/div/div/button')
    ]

    for selector in modal_close_selectors:
        try:
            close_button = wait.until(EC.element_to_be_clickable(selector))
            close_button.click()
            time.sleep(2)  # Wait for modal to close and next modal to appear
        except TimeoutException:
            print("Modal not found or already closed")


# Close modals
close_modals()


# Function to scroll the page gradually
def gradual_scroll():
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down by a small amount
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(0.5)  # Wait for content to load

        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break

        last_height = new_height


# Function to click "Show More Products" until all products are loaded
def load_all_products():
    while True:
        try:
            print('scrolling to bottom')
            # Gradually scroll to the bottom of the page
            gradual_scroll()

            # Find and click the "Show More Products" button
            show_more_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.css-1p9axos.eanm77i0")))
            show_more_button.click()

            # Wait for new products to load
            time.sleep(1)
        except selenium.common.exceptions.ElementClickInterceptedException:
            close_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div/div/div[2]/div/div/button')))
            close_button.click()
            time.sleep(0.5)
        except TimeoutException:
            if not element_exists('/html/body/div[5]/div/div/div[2]/div/div/button'):
                # No more "Show More Products" button to click
                print('No more "Show More Products" button to click')
                break

            close_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div/div/div[2]/div/div/button')))
            close_button.click()
            time.sleep(0.5)


def element_exists(xpath):
    try:
        driver.find_element(By.XPATH, xpath)
        return True
    except selenium.common.exceptions.NoSuchElementException:
        return False


# Perform a final gradual scroll to the bottom to ensure all products are loaded
gradual_scroll()

# Load all products
load_all_products()

# Find all product links
product_elements = driver.find_elements(By.CSS_SELECTOR, "a.css-klx76")

# Extract href attributes
product_links = [element.get_attribute('href') for element in product_elements]

print(len(product_links))

# Dump product links into a JSON file
with open('products.json', 'w') as json_file:
    json.dump(product_links, json_file, indent=4)

# Close the webdriver
driver.quit()
