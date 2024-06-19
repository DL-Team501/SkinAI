import json
import re

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
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

# Load product links from JSON file
with open('products.json', 'r') as json_file:
    product_links = json.load(json_file)

is_closed = {'v': False}


# Function to close modals
def close_modals():
    if is_closed['v']:
        return

    modal_close_selectors = [
        (By.CSS_SELECTOR, "button.css-1kna575[data-at='modal_close']"),
        (By.CSS_SELECTOR, "button.css-1kna575[data-at='close_button']")
    ]

    for selector in modal_close_selectors:
        try:
            close_button = wait.until(EC.element_to_be_clickable(selector))
            close_button.click()
            time.sleep(0.5)  # Wait for modal to close and next modal to appear
        except TimeoutException:
            print("Modal not found or already closed")

    is_closed['v'] = True


def extract_skincare_concerns(text):
    """Extracts the skin type from a text string using regular expressions."""

    # Pattern Explanation:
    #  - `Skin Type:\s*` : Matches "Skin Type:" followed by zero or more whitespace characters.
    #  - `(.+)`         : Captures one or more characters (the actual skin type) into a group.
    #  - `\n`           : Matches the newline character.
    pattern = r"Skincare Concerns:\s*(.+)\n"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        return match.group(1)  # Return the captured skin type (group 1)
    else:
        return "Not Found"


def extract_skin_type(text):
    """Extracts the skin type from a text string using regular expressions."""

    # Pattern Explanation:
    #  - `Skin Type:\s*` : Matches "Skin Type:" followed by zero or more whitespace characters.
    #  - `(.+)`         : Captures one or more characters (the actual skin type) into a group.
    #  - `\n`           : Matches the newline character.
    pattern = r"Skin Type:\s*(.+)\n"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        return match.group(1)  # Return the captured skin type (group 1)
    else:
        return "Not Found"


# Function to scrape ingredients, skin type, and skincare concerns
def scrape_product_data(product_link):
    driver.get(product_link)

    # Close modals
    close_modals()

    # Initialize variables
    ingredients_text = skin_type_text = skincare_concerns_text = "Not Found"

    # Wait for the ingredients element to load
    try:
        ingredients_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-at='ingredients']")))
        ingredients_button.click()
        ingredients_element = wait.until(EC.presence_of_element_located((By.ID, "ingredients")))
        ingredients_text = ingredients_element.text.split('\n\n')[1]
    except TimeoutException:
        print("Timed out waiting for ingredients element to load")
    except Exception as e:
        print(f"Error while scraping ingredients: {str(e)}")
    try:
        product_about_text = driver.find_element(By.XPATH, "/html/body/div[2]/main/section/div[6]/div[2]/div").text
    except Exception as e:
        print(f"Error while scraping product about text: {str(e)}")

    # Wait for the skin type element to load
    try:
        skin_type_text = extract_skin_type(product_about_text)
    except Exception as e:
        print(f"Error while scraping skin type: {str(e)}")

    # Wait for the skincare concerns element to load
    try:
        skincare_concerns_text = extract_skincare_concerns(product_about_text)
    except Exception as e:
        print(f"Error while scraping skincare concerns: {str(e)}")

    return {
        "Ingredients": ingredients_text,
        "Skin Type": skin_type_text,
        "Skincare Concerns": skincare_concerns_text
    }


# Scrape data for each product link
product_data = {}
for index, product_link in enumerate(product_links, start=1):
    print(f"Scraping data for product {index}/{len(product_links)}")
    product_data[product_link] = scrape_product_data(product_link)
    print(product_link)
    print(json.dumps(product_data[product_link], indent=2))
    time.sleep(0.5)  # Add a short delay between requests to avoid being rate-limited

# Save scraped data to JSON file
with open('product_data.json', 'w') as json_file:
    json.dump(product_data, json_file, indent=4)

# Close the webdriver
driver.quit()
