import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('survey.csv').drop(columns=['Timestamp', 'comments', 'state'])

# Age
age_mode = df.Age.mode()[0]
df.loc[(df.Age < 18) | (df.Age > 100), 'Age'] = age_mode

# Gender
male_values = {'male', 'm', 'make', 'cis male', 'man', 'msle', 'mail', 'malr', 'cis man', 'male (cis)', 'maile', 'mal'}
female_values = {'female', 'f', 'woman', 'female (cis)', 'femail', 'cis female', 'femake'}
df.Gender = df.Gender.str.lower().str.strip()
df.loc[df.Gender.isin(male_values), 'Sex'] = 'Mal'
df.loc[df.Gender.isin(female_values), 'Sex'] = 'Female'
df.loc[~df.Gender.isin(female_values) & ~df.Gender.isin(male_values), 'Sex'] = 'Other'
df.drop(columns='Gender', inplace=True)

# Country
df.loc[df.Country != 'United States', 'Country'] = 'Other'

df.work_interfere = df.work_interfere.map({'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3})
df.no_employees = df.no_employees.map(
    {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5}
)
df.leave = df.leave.map(
    {'Very difficult': 0, 'Somewhat difficult': 1, "Don't know": 2, 'Somewhat easy': 3, 'Very easy': 4}
)
for col in [
    'benefits', 'wellness_program', 'seek_help', 'anonymity', 'care_options', 'mental_health_consequence',
    'phys_health_consequence', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical',
    'obs_consequence', 'coworkers'
]:
    df[col] = df[col].map({
        'No': 0,
        "Don't know": 1, 'Not sure': 1, 'Maybe': 1, 'Some of them': 1,
        'Yes': 2
    })

df = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(df), columns=df.columns)

for col in ['Country', 'self_employed', 'family_history', 'treatment', 'remote_work', 'tech_company', 'Sex']:
    df[col] = LabelEncoder().fit_transform(df[col])

clf = HistGradientBoostingClassifier(random_state=0)
cv = StratifiedShuffleSplit(random_state=0)
for col in df.columns:
    if col == 'Age':
        continue
    X, y = df.drop(columns=col), df[col].astype('int')
    score = cross_val_score(clf, X, y, scoring='balanced_accuracy', cv=cv).mean()
    print(f"{col} - Balanced accuracy:{score:f}")

'''
Country - Balanced accuracy:0.700706
self_employed - Balanced accuracy:0.731081
family_history - Balanced accuracy:0.624583
treatment - Balanced accuracy:0.742641
work_interfere - Balanced accuracy:0.291160
no_employees - Balanced accuracy:0.322369
remote_work - Balanced accuracy:0.617315
tech_company - Balanced accuracy:0.567265
benefits - Balanced accuracy:0.696187
care_options - Balanced accuracy:0.594987
wellness_program - Balanced accuracy:0.619448
seek_help - Balanced accuracy:0.651627
anonymity - Balanced accuracy:0.523570
leave - Balanced accuracy:0.327407
mental_health_consequence - Balanced accuracy:0.646787
phys_health_consequence - Balanced accuracy:0.495121
coworkers - Balanced accuracy:0.500982
supervisor - Balanced accuracy:0.588010
mental_health_interview - Balanced accuracy:0.561237
phys_health_interview - Balanced accuracy:0.503595
mental_vs_physical - Balanced accuracy:0.555443
obs_consequence - Balanced accuracy:0.594907
Sex - Balanced accuracy:0.413717
'''
