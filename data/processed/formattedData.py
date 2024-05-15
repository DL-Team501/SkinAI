import re
import pandas as pd
import numpy as np

# formattedData

def getFormattedData():
    df = pd.read_csv('../raw/cosmetic.csv')

    # Apply the cleaning function to the 'ingredients list' column
    df['clean_ingredients'] = df['ingredients'].apply(clean_ingredients)
    unique_ingredients = getUniqueIngredients(df['clean_ingredients'])
    ingredient_index_dict = {ingredient: index + 1 for index, ingredient in enumerate(unique_ingredients)}

    df['ingredient_index'] = df['clean_ingredients'].apply(lambda row: map_ingredient_to_index(row, ingredient_index_dict))
    max_ingredients_list_length = max(map(len, df['ingredient_index']))

    # Pad lists in the 'ingredient_index' column with zeros to make their length 200
    df['ingredient_index'] = df['ingredient_index'].apply(lambda row: pad_list_with_zeros(row, max_ingredients_list_length))

    # Apply positional encoding to each vector in the 'ingredients_index' column
    df['ingredients_index_encoded'] = df['ingredient_index'].apply(apply_positional_encoding)

    skin_type_cols = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    df_skin_types = df[skin_type_cols]
    df.loc[(df_skin_types == 0).all(axis=1), 'Normal'] = 1
    df['skin_types_list'] = df.apply(lambda row: [row[col] for col in skin_type_cols], axis=1)
    clean_dataframe(df)

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

def getUniqueIngredients(clean_ingredients_lists):
    unique_ingredients = set()
    for ingredient_list in clean_ingredients_lists:
        ingredients = ingredient_list

        unique_ingredients.update(ingredient.strip() for ingredient in ingredients)

    len(unique_ingredients)

    unique_ingredients = list(unique_ingredients)
    unique_ingredients = [a for a in unique_ingredients if a]

    # Define a function to map ingredients to their indices

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

def positional_encoding(max_len, d_model):
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def apply_positional_encoding(row):
    max_len = len(row)
    d_model = 1  # Dimensionality of the vectors
    pe = positional_encoding(max_len, d_model)
    row = np.array(row)  # Convert the list to a numpy array for easier manipulation
    try:
      return row + pe
    except TypeError as e:
      print("Error occurred at row:", row)
      raise e

def clean_dataframe(df): 
    df = df.drop('ingredient_index', axis=1)
    df = df.drop('Combination', axis=1)
    df = df.drop('Dry', axis=1)
    df = df.drop('Normal', axis=1)
    df = df.drop('Sensitive', axis=1)
    df = df.drop('Oily', axis=1)
    df = df.drop('price', axis=1)
    df = df.drop('ingredients', axis=1)

