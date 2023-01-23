import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

df = pd.read_csv('cereal.csv').drop(columns='type')
df = df[df.mfr.isin(['K', 'G'])]
X, y = df.drop(columns='mfr'), LabelEncoder().fit_transform(df.mfr)
clf = SVC(kernel='linear', random_state=0)
score = cross_val_score(clf, X, y, scoring='balanced_accuracy').mean()
print(f"Manufacturer - Balanced accuracy:{score:f}")
