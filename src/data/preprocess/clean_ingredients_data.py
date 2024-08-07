import os
import re
from typing import List

import pandas as pd

from src.consts import ROOT_PATH

skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']


def get_sephora_csv() -> pd.DataFrame:
    data_file = os.path.join(ROOT_PATH, 'data', 'raw', 'sephora.csv')
    df = pd.read_csv(data_file)
    df = filter_no_ingredients(df)
    df['clean_ingredients'] = df['Ingredient List'].apply(clean_ingredients)
    df['Skincare Concerns'] = df['Skincare Concerns'].apply(handle_and)
    df['Skin Types'] = df['Skin Types'].apply(handle_and)
    df['Skincare Concerns'] = df['Skincare Concerns'].apply(strip_strings)
    df['Skin Types'] = df['Skin Types'].apply(strip_strings)
    df['skin_types_list'] = df['Skin Types'].apply(lambda x: create_binary_list(x, skin_types))

    return df


def handle_and(strings: List[str]) -> List[str]:
    processed_strings = []

    for s in strings:
        if s.startswith('and') or s[1:4] == 'and':
            curr_processed_strings = [s.replace('and', '', 1)]
        else:
            curr_processed_strings = s.split('and')

        processed_strings.extend(curr_processed_strings)

    return processed_strings


def strip_strings(strings: List[str]) -> List[str]:
    return [s.strip() for s in strings]


def create_binary_list(input_list, target_strings):
    return [1 if target in input_list else 0 for target in target_strings]


def filter_no_ingredients(df):
    return df[df['Ingredient List'].apply(lambda x: len(x) > 0)]


def get_cosmetic_csv() -> pd.DataFrame:
    data_file = os.path.join(ROOT_PATH, 'data', 'raw', 'cosmetic.csv')
    df = pd.read_csv(data_file)
    df = filter_bad_rows(df)
    df['clean_ingredients'] = df['ingredients'].apply(clean_ingredients)

    df_skin_types = df[skin_types]
    df.loc[(df_skin_types == 0).all(axis=1), 'Normal'] = 1

    df['skin_types_list'] = df.apply(lambda row: [row[col] for col in skin_types], axis=1)
    df = df.drop(columns=['rank', 'price', 'ingredients'] + skin_types)

    return df


def get_all_data_df() -> pd.DataFrame:
    columns_to_keep = ['clean_ingredients', 'skin_types_list']

    cosmetic_df = get_cosmetic_csv()[columns_to_keep]
    sephora_df = get_sephora_csv()[columns_to_keep]

    df = pd.concat([cosmetic_df, sephora_df], ignore_index=True)

    return df


def get_formatted_data():
    df = get_all_data_df()

    # Transform the ingredients list to indexes list
    unique_ingredients = get_unique_ingredients(df['clean_ingredients'])
    ingredient_index_dict = {ingredient: index + 1 for index, ingredient in enumerate(unique_ingredients)}

    df['tokenized_ingredients'] = df['clean_ingredients'].apply(
        lambda row: map_ingredient_to_index(row, ingredient_index_dict))
    max_ingredients_list_length = max(map(len, df['tokenized_ingredients']))

    # Pad lists in the 'tokenized_ingredients' column with zeros to make their length equal
    df['tokenized_ingredients'] = df['tokenized_ingredients'].apply(
        lambda row: pad_list_with_zeros(row, max_ingredients_list_length))

    return df


def filter_bad_rows(df):
    df = df[~df['ingredients'].astype(str).str.startswith("Visit")]
    df = df[~df['ingredients'].astype(str).str.startswith("No Info")]
    mask = df['ingredients'].astype(str).str.endswith("for the most up to date list of ingredients.")
    df.loc[mask, 'ingredients'] = df.loc[mask, 'ingredients'].astype(str).apply(remove_after_newline)
    df = df[df['ingredients'].astype(str).str.contains(',')]

    return df


def remove_after_newline(text):
    last_newline_index = text.rfind('\n')
    if last_newline_index != -1:
        return text[:last_newline_index].rstrip('\n')
    else:
        return text


def clean_ingredients(text):
    """
  Cleans the ingredient text.

  Args:
      text: String containing the ingredient list.

  Returns:
      A string with cleaned ingredients in lowercase, with extra spaces removed.
  """
    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    ingredient_list = [ingredient.strip() for ingredient in text.split(', ')]
    ingredient_list[-1] = ingredient_list[-1].rstrip('. ')
    ingredient_list[-1] = ingredient_list[-1].rstrip('.')
    ingredient_list = [item for item in ingredient_list if item != ""]

    return ingredient_list


def get_unique_ingredients(clean_ingredients_lists):
    unique_ingredients = set()

    for ingredient_list in clean_ingredients_lists:
        ingredients = ingredient_list
        unique_ingredients.update(ingredient.strip() for ingredient in ingredients)

    print(f"Number of unique ingredients: {len(unique_ingredients)}")

    unique_ingredients = list(unique_ingredients)
    unique_ingredients.append('<UNK>')

    return unique_ingredients


def map_ingredient_to_index(ingredient_list, ingredient_index_dict):
    ingredient_list_indexes = []
    for ingredient in ingredient_list:
        if ingredient not in ingredient_index_dict.keys():
            print(f'NOT FOUND: "{ingredient}"')
        ingredient_list_indexes.append(ingredient_index_dict.get(ingredient))

    return ingredient_list_indexes


def pad_list_with_zeros(lst, length):
    if len(lst) >= length:
        return lst[:length]
    else:
        return lst + [0] * (length - len(lst))
