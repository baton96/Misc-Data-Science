import json
import re

import inflect
import unidecode

singularize = inflect.engine().singular_noun

with open('ingredients.json', 'r') as fin, open('ingredients.csv', 'w') as fout:
    fout.write('cuisine,ingredients\n')
    recipes = json.load(fin)
    for recipe in recipes:
        ingredients = set()
        for ingredient in recipe['ingredients']:
            ingredient = re.sub(r'\d', ' ', ingredient)
            ingredient = re.sub(r'\(.*\)', ' ', ingredient)
            ingredient = ingredient.split(',')[0]
            ingredient = unidecode.unidecode(ingredient)
            ingredient = re.sub(r'[^\w\s]', ' ', ingredient)
            ingredient = ingredient.lower().strip()
            ingredient = re.sub(r' +', ' ', ingredient)
            ings = []
            for ing in ingredient.split(' '):
                ing = singularize(ing) or ing
                ingredients.add(ing)
        fout.write(f"{recipe['cuisine']},{' '.join(ingredients)}\n")
