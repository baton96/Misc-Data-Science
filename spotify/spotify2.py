import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

df = pd.read_csv('spotify2.csv')
df = df[(df['artist type'] == 'Solo') | (df['artist type'] == 'Band/Group')]

df['top genre'] = df['top genre'].fillna('')
df['radio'] = df.title.str.contains('Radio', regex=False)
# df['has_feat'] = df.title.str.contains('feat.', regex=False)
for genre in ['dance', 'hip hop', 'pop']:
    df[genre] = df['top genre'].str.contains(genre, regex=False)

X, y = df.drop(columns=['title', 'artist', 'top genre', 'added', 'artist type']), df['artist type']
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=42)
score = cross_val_score(
    clf,
    X,
    y,
    scoring='balanced_accuracy',
    cv=StratifiedShuffleSplit(random_state=0)
).mean()
print(f"Balanced accuracy:{score:f}")  # 0.681608
