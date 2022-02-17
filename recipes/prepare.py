import json

import pandas as pd

recipes = []
with open('recipes.json', 'r') as f:
    for line in f:
        recipe = {}
        item = json.loads(line)
        _recipe = item['page']['recipe']
        for field in ['skill_level', 'cooking_time', 'prep_time', 'serves']:
            recipe[field] = _recipe[field]
        for nutrient_amount in _recipe['nutrition_info']:
            nutrient, amount = nutrient_amount.split(' ')
            recipe[nutrient.lower()] = amount.rstrip('g')
        recipe.setdefault('sugar', None)
        recipe['ingredients'] = ' '.join(_recipe['ingredients']).replace(',', '')
        recipes.append(recipe)

df = pd.DataFrame(recipes)
df.to_csv('recipes.csv')
