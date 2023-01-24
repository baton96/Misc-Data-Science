import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('jobs.csv').drop(
    columns=['Job Title', 'Job Descriptions', 'Company Sector', 'Company Industry', 'State']
)

df['Melbourne'] = df['Job Location'] == 'Melbourne'
df['Sydney'] = df['Job Location'] == 'Sydney'
df.drop(columns='Job Location', inplace=True)

df['Private'] = df['Company Type'] == 'Company - Private'
df['Public'] = df['Company Type'] == 'Company - Public'
df.drop(columns='Company Type', inplace=True)

df['Company Revenue'] = LabelEncoder().fit_transform(df['Company Revenue'].fillna('Unknown / Non-Applicable'))
df['Company Size'] = LabelEncoder().fit_transform(df['Company Size'])

cv = StratifiedShuffleSplit(random_state=0)
clf = HistGradientBoostingClassifier(random_state=0)
# for col in ['Melbourne', 'Sydney', 'Private', 'Public', 'Company Size', 'Company Revenue']:
yn_cols = [col for col in df.columns if '_yn' in col]
for col in yn_cols:
    X, y = df.drop(columns=col), df[col]
    try:
        score = cross_val_score(clf, X, y, scoring='balanced_accuracy', cv=cv).mean()
        print(f"{col} - Balanced accuracy:{score:f}")
    except:
        pass

'''
Melbourne - Balanced accuracy:0.964620
Sydney - Balanced accuracy:0.981418
Private - Balanced accuracy:0.985863
Public - Balanced accuracy:0.983313
Company Size - Balanced accuracy:0.911035
Company Revenue - Balanced accuracy:0.900228
'''
