import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('students.csv')
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])
for col in df.columns:
    X, y = df.drop(columns=col), df[col]
    clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=42)
    score = cross_val_score(clf, X, y, scoring='balanced_accuracy').mean()
    print(f"{col} - Balanced accuracy:{score:f}")
'''
Gender - Balanced accuracy:0.865369
Age - Balanced accuracy:0.889087
Education Level - Balanced accuracy:0.990842
Institution Type - Balanced accuracy:0.964459
IT Student - Balanced accuracy:0.941424
Location - Balanced accuracy:0.925084
Load-shedding - Balanced accuracy:0.941738
Financial Condition - Balanced accuracy:0.863140
Internet Type - Balanced accuracy:0.944689
Network Type - Balanced accuracy:0.730024
Class Duration - Balanced accuracy:0.897922
Self Lms - Balanced accuracy:0.961438
Device - Balanced accuracy:0.879865
Adaptivity Level - Balanced accuracy:0.853156
'''
