import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("train.csv")
y = X['Survived']

X['Family_Size'] = X['Parch'] + X['SibSp']
X['Sex'] = X['Sex'] == 'male'

X['Title'] = X['Name'].str.extract(r'([A-Za-z]+)\.')
mapping = {
    'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
    'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'
}
X.replace({'Title': mapping}, inplace=True)
for title in list(X['Title'].unique()):
    mean_age_per_title = X.groupby('Title')['Age'].mean()[title]
    idxs = (X['Age'].isnull()) & (X['Title'] == title)
    X.loc[idxs, 'Age'] = mean_age_per_title
X.loc[X['Fare'].isnull(), 'Fare'] = X['Fare'].mean()
X['FareBin'] = pd.qcut(X['Fare'], 5).cat.codes
X['AgeBin'] = pd.qcut(X['Age'], 4).cat.codes

X['Last_Name'] = X['Name'].apply(lambda x: str.split(x, ",")[0])
X['Family_Survival'] = 0.5

for _, group in X.groupby(['Last_Name', 'Fare']):
    if len(group) <= 1:
        continue
    family_survival = int(group['Survived'].mean() > 0.5)
    X.loc[group.index, 'Family_Survival'] = family_survival

for _, group in X.groupby('Ticket'):
    if len(group) <= 1:
        continue
    for idx, row in group.iterrows():
        if row['Family_Survival'] > 0.5:
            continue
        family_survival = int(group.drop(idx)['Survived'].mean() > 0.5)
        idxs = X['PassengerId'] == row['PassengerId']
        X.loc[idxs, 'Family_Survival'] = family_survival

X = X[['Pclass', 'Sex', 'Family_Size', 'Family_Survival', 'FareBin', 'AgeBin']]

pipeline = make_pipeline(
    SimpleImputer(), 
    StandardScaler(), 
    RandomForestClassifier(n_jobs=-1, random_state=1),
    # NeighborhoodComponentsAnalysis(n_components=2, random_state=0),
)
'''
X2 = pipeline.fit_transform(X, y)
plt.scatter(X2[:, 0], X2[:, 1], c=y)
plt.show()
'''
print(
    "Cross-validation accuracy:%f" % cross_val_score(
        pipeline,
        X,
        y,
        scoring='balanced_accuracy',
        cv=StratifiedShuffleSplit(random_state=0)
    ).mean()
)

