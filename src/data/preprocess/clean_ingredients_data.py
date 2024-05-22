import os
import re
import math
import torch
import pandas as pd

from src.consts import ROOT_PATH


def get_formatted_data():
    data_file = os.path.join(ROOT_PATH, 'data', 'raw', 'cosmetic.csv')
    df = pd.read_csv(data_file)

    # Apply the cleaning function to the 'ingredients list' column
    df['clean_ingredients'] = df['ingredients'].apply(clean_ingredients)

    # Transform the ingredients list to indexes list
    unique_ingredients = get_unique_ingredients(df['clean_ingredients'])
    ingredient_index_dict = {ingredient: index + 1 for index, ingredient in enumerate(unique_ingredients)}
    df['ingredients_indexed'] = df['clean_ingredients'].apply(
        lambda row: map_ingredient_to_index(row, ingredient_index_dict))
    max_ingredients_list_length = max(map(len, df['ingredients_indexed']))

    # Pad lists in the 'ingredients_indexed' column with zeros to make their length equal
    df['ingredients_indexed'] = df['ingredients_indexed'].apply(
        lambda row: pad_list_with_zeros(row, max_ingredients_list_length))

    # Apply positional encoding to each vector in the 'ingredients_index' column
    df['ingredients_index_encoded'] = df['ingredients_indexed'].apply(apply_positional_encoding)

    # Put 1 in normal skin type column if all the skin types are 0
    skin_type_cols = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    df_skin_types = df[skin_type_cols]
    df.loc[(df_skin_types == 0).all(axis=1), 'Normal'] = 1

    df['skin_types_list'] = df.apply(lambda row: [row[col] for col in skin_type_cols], axis=1)
    df = df.drop(columns=['rank', 'ingredients_indexed', 'price', 'ingredients'] + skin_type_cols)

    return df


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
    # unique_ingredients = [a for a in unique_ingredients if a]

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


def get_positional_encoding(max_seq_len, d_model):
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def apply_positional_encoding(ingredients_list):
    d_model = 1
    max_len = len(ingredients_list)
    pe = get_positional_encoding(max_len, d_model).squeeze()
    ingredients_tensor = torch.tensor(ingredients_list)

    return ingredients_tensor + pe
