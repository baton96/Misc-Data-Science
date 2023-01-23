import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('students.csv')
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])
for col in df.columns:
    X, y = df.drop(columns=col), df[col]
    clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=42)
    score = cross_val_score(clf, X, y, scoring='balanced_accuracy', cv=StratifiedShuffleSplit(random_state=0)).mean()
    print(f"{col} - Balanced accuracy:{score:f}")
'''
Gender - Balanced accuracy:0.869638
Age - Balanced accuracy:0.898039
Education Level - Balanced accuracy:0.994340
Institution Type - Balanced accuracy:0.959956
IT Student - Balanced accuracy:0.957760
Location - Balanced accuracy:0.944996
Load-shedding - Balanced accuracy:0.970767
Financial Condition - Balanced accuracy:0.861658
Internet Type - Balanced accuracy:0.952003
Network Type - Balanced accuracy:0.785168
Class Duration - Balanced accuracy:0.920238
Self Lms - Balanced accuracy:0.962238
Device - Balanced accuracy:0.875490
Adaptivity Level - Balanced accuracy:0.865245
'''
