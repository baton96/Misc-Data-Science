import pandas as pd

df = pd.read_csv('df.csv')
colors = df.Colors

for i in df.index:
    colors = df.at[i, 'Colors']
    hex_colors = [color.strip("'#") for color in colors[1:-1].split(', ')]
    r, g, b = zip(*[tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) for color in hex_colors])
    r, g, b = sum(r), sum(g), sum(b)
    df.at[i, 'r'] = int(r)
    df.at[i, 'g'] = int(g)
    df.at[i, 'b'] = int(b)
df.drop(['Colors', 'Link'], axis=1).to_csv('gogh.csv')
