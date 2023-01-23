import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('spotify1.csv')
df.genre = df.genre.fillna('')
df['radio'] = df.song.str.contains('Radio', regex=False)
# df['has_feat'] = df.song.str.contains('feat.', regex=False)

genres = set()
for genre in df.genre:
    genres.update(genre.split(', '))
genres.discard('')
for genre in genres:
    df[genre] = df.genre.str.contains(genre, regex=False)

X, y = df.drop(columns=['artist', 'song', 'genre', 'explicit']), df.explicit
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=42)
score = cross_val_score(clf, X, y, scoring='balanced_accuracy').mean()
print(f"Balanced accuracy:{score:f}")
