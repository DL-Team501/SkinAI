import json
import time
from json import JSONDecodeError

import pydash
import requests

is_products_empty = False
sephora_base_url = "https://www.sephora.com"

product_links = []
current_page = 1

while not is_products_empty:

    response = requests.get('https://sephora.p.rapidapi.com/us/products/v2/list',
                            headers={"x-rapidapi-host": "sephora.p.rapidapi.com",
                                     "x-rapidapi-key": "ead2eb722amsh72549df50a19bc5p16bd69jsnad8ebf09cce1"},
                            params={'pageSize': 60, 'currentPage': current_page, 'categoryId': 'cat150006'})
    try:
        response_data = response.json()

        products_info = response_data['products']

        print(len(products_info))
        if len(products_info) > 0:
            product_links.extend(
                list(map(lambda curr_product: {"url": sephora_base_url + curr_product['targetUrl'],
                                               "id": curr_product['productId'],
                                               "skuId": pydash.get(curr_product, 'currentSku.skuId')}, products_info)))
        else:
            is_products_empty = True

        time.sleep(0.2)
        current_page += 1
    except JSONDecodeError:
        print(response.text)
        time.sleep(1)

with open('products.json', 'w') as json_file:
    json.dump(product_links, json_file, indent=4)
