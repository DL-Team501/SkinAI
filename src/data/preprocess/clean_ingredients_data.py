import os
import re
import ast
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.consts import ROOT_PATH, INGREDIENT_LIST_CLASSIFICATION_LABELS


@dataclass
class SkinCareData:
    data: pd.DataFrame
    ingredient_index_dict: Dict


def get_formatted_data() -> SkinCareData:
    df = load_old_sephora_csv_to_df()

    # Transform the ingredients list to indexes list
    unique_ingredients = get_unique_ingredients(df['clean_ingredients'])
    ingredient_index_dict = {ingredient: index for index, ingredient in enumerate(unique_ingredients)}

    df['tokenized_ingredients'] = df['clean_ingredients'].apply(
        lambda row: map_ingredient_to_index(row, ingredient_index_dict))
    max_ingredients_list_length = max(map(len, df['tokenized_ingredients']))

    # Pad lists in the 'tokenized_ingredients' column with zeros to make their length equal
    df['tokenized_ingredients'] = df['tokenized_ingredients'].apply(
        lambda row: pad_list_with_zeros(row, max_ingredients_list_length))

    return SkinCareData(data=df, ingredient_index_dict=ingredient_index_dict)


def load_old_sephora_csv_to_df() -> pd.DataFrame:
    data_file = os.path.join(ROOT_PATH, 'data', 'raw', 'cosmetic.csv')
    df = pd.read_csv(data_file)
    df['clean_ingredients'] = df['ingredients'].str.split(', ')
    df['one_hot_labels'] = df[['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']].values.tolist()
    columns_to_keep = ['clean_ingredients', 'one_hot_labels']

    return df[columns_to_keep]


def load_sephora_csv_to_df() -> pd.DataFrame:
    data_file = os.path.join(ROOT_PATH, 'data', 'raw', 'sephora_data_clean.csv')
    df = pd.read_csv(data_file)
    df = filter_no_ingredients(df)
    df['clean_ingredients'] = df['Ingredient List'].apply(clean_ingredients)
    one_hot_labels_df = df['Skin Types and Concerns'].apply(one_hot_labels)
    df = pd.concat([df, one_hot_labels_df], axis=1)
    df['one_hot_labels'] = df[INGREDIENT_LIST_CLASSIFICATION_LABELS].apply(lambda row: row.tolist(), axis=1)
    columns_to_keep = ['clean_ingredients', 'one_hot_labels']

    return df[columns_to_keep]


def one_hot_labels(skin_types_concerns):
    if isinstance(skin_types_concerns, str):
        skin_types_concerns = ast.literal_eval(skin_types_concerns)

    vector = {cls: 0 for cls in INGREDIENT_LIST_CLASSIFICATION_LABELS}
    skin_types_concerns_lower = [item.lower() for item in skin_types_concerns]

    for cls in INGREDIENT_LIST_CLASSIFICATION_LABELS:
        if any(cls in item for item in skin_types_concerns_lower):
            vector[cls] = 1

    return pd.Series(vector)


def filter_no_ingredients(df):
    return df[df['Ingredient List'].apply(lambda x: len(x) > 0)]


def clean_ingredients(ingredient_list):
    # If the ingredient_list is already a list, clean it directly
    if isinstance(ingredient_list, list):
        ingredients = ingredient_list
    # If it's a string, convert to a list first
    elif isinstance(ingredient_list, str):
        ingredients = ast.literal_eval(ingredient_list)
    else:
        print("No list")
        # Handle cases where the ingredient list is not a list or string (e.g., NaN)
        return []

    # Remove text after '-', remove special characters, and strip whitespace
    ingredients = [re.sub(r'^-.*:', '', ing).strip().lower() for ing in ingredients]
    ingredients = [re.sub(r'[^a-zA-Z0-9\s]', '', ing).strip() for ing in ingredients]

    return ingredients


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
        ingredient_list_indexes.append(ingredient_index_dict.get(ingredient, 0))

    return ingredient_list_indexes


def pad_list_with_zeros(lst, length):
    if len(lst) >= length:
        return lst[:length]
    else:
        return lst + [0] * (length - len(lst))
