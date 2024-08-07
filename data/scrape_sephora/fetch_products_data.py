import json
import math
import time
from json import JSONDecodeError

import pydash
import requests

with open('products.json', 'r') as json_file:
    products_metadata = json.load(json_file)

start_index = 2700


def get_product_data(curr_index, product_metadata):

    response = requests.get('https://sephora.p.rapidapi.com/us/products/v2/detail',
                            headers={"x-rapidapi-host": "sephora.p.rapidapi.com",
                                     "x-rapidapi-key": "8ec274ab7cmsh7bd26823d4a26d7p13e71ajsn0ee25cf04fa2"},
                            params={'productId': product_metadata['id'], 'preferedSku': product_metadata['skuId']})

    try:
        print(response.text)
        print(curr_index)
        response_data = response.json()
        product_data = {'displayName': pydash.get(response_data, 'productDetails.displayName'),
                        'longDescription': pydash.get(response_data, 'productDetails.longDescription'),
                        'shortDescription': pydash.get(response_data, 'productDetails.shortDescription'),
                        'suggestedUsage': pydash.get(response_data, 'productDetails.suggestedUsage'),
                        'ingredientDesc': pydash.get(response_data, 'currentSku.ingredientDesc')}

        return pydash.merge(product_data, product_metadata)
    except JSONDecodeError:
        print(response.text)
        time.sleep(1)

        return get_product_data(curr_index, product_metadata)


products_data = list(
    map(lambda indexed_product_metadata: get_product_data(indexed_product_metadata[0], indexed_product_metadata[1]),
        enumerate(products_metadata[start_index:min(start_index + 450, len(products_metadata))])))

with open('products_data-7.json', 'w') as json_file:
    json.dump(products_data, json_file, indent=4)
