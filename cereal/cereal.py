import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('cereal.csv').drop(columns='type')
df = df[df.mfr.isin(['K', 'G'])]
X, y = df.drop(columns='mfr'), LabelEncoder().fit_transform(df.mfr)
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=42)
score = cross_val_score(clf, X, y, scoring='balanced_accuracy').mean()
print(f"Manufacturer - Balanced accuracy:{score:f}")
