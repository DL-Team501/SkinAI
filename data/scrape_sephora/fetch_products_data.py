import json
import time
from json import JSONDecodeError

import pydash
import requests

with open('products.json', 'r') as json_file:
    products_metadata = json.load(json_file)

counter = 0


def get_product_data(product_metadata):
    global counter

    response = requests.get('https://sephora.p.rapidapi.com/us/products/v2/detail',
                            headers={"x-rapidapi-host": "sephora.p.rapidapi.com",
                                     "x-rapidapi-key": "ead2eb722amsh72549df50a19bc5p16bd69jsnad8ebf09cce1"},
                            params={'productId': product_metadata['id'], 'preferedSku': product_metadata['skuId']})

    try:
        print(counter)
        print(response.text)
        response_data = response.json()
        product_data = {'displayName': pydash.get(response_data, 'productDetails.displayName'),
                        'longDescription': pydash.get(response_data, 'productDetails.longDescription'),
                        'shortDescription': pydash.get(response_data, 'productDetails.shortDescription'),
                        'suggestedUsage': pydash.get(response_data, 'productDetails.suggestedUsage'),
                        'ingredientDesc': pydash.get(response_data, 'currentSku.ingredientDesc')}
        counter += 1

        return pydash.merge(product_data, product_metadata)
    except JSONDecodeError:
        print(response.text)
        time.sleep(1)

        return get_product_data(product_metadata)


products_data = list(map(lambda product_metadata: get_product_data(product_metadata), products_metadata))

with open('products_data.json', 'w') as json_file:
    json.dump(products_data, json_file, indent=4)
