class CosmeticIngredientTokenizer:
    def __init__(self, ingredient_dict):
        self.ingredient_dict = ingredient_dict
        self.unk_token = "<UNK>"  # Token for unknown ingredients

    def tokenize(self, ingredient_list):
        token_ids = [self.ingredient_dict.get(ingredient, self.ingredient_dict[self.unk_token])
                     for ingredient in ingredient_list]

        return token_ids
