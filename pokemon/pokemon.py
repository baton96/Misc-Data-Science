import warnings

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

df = pd.read_csv('pokemon.csv')
for target in ['legendary', 'type1']:
    if target == 'legendary':
        X, y = df.drop(['name', 'legendary'], axis=1), df.legendary
        X = pd.get_dummies(X, columns=['type1', 'type2'])
        clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=15)
        score = cross_val_score(clf, X, y, scoring='balanced_accuracy').mean()
        print(f"Target: {target}, balanced accuracy:{score:f}")  # 0.924910
    elif target == 'type1':
        X, y = df.name, df.type1
        pipeline = make_pipeline(
            TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), smooth_idf=False, sublinear_tf=True),
            FunctionTransformer(lambda x: x.todense()),
            LinearSVC(dual=False)
        )
        score = cross_val_score(pipeline, X, y, scoring='balanced_accuracy').mean()
        print(f"Target: {target}, balanced accuracy:{score:f}")  # 0.217263
