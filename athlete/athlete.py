import pandas as pd
df = pd.read_csv('athlete.csv')
for col in ['Sex', 'Season', 'Medal']:
    print(df[col].value_counts(normalize=True, dropna=False), '\n')
