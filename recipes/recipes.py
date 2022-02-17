import warnings

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore")


class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=-1, random_state=1):
        self.n_jobs, self.random_state = n_jobs, random_state
        self.tab_clf = BalancedRandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
        self.text_clf = make_pipeline(
            TfidfVectorizer(),
            FunctionTransformer(lambda x: x.todense()),
            BalancedRandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
        )

    def fit(self, X, y):
        X_tab, X_text = X.drop('ingredients', axis=1), X.ingredients
        self.tab_clf.fit(X_tab, y)
        self.text_clf.fit(X_text, y)
        return self

    def predict(self, X):
        X_tab, X_text = X.drop('ingredients', axis=1), X.ingredients
        tab_proba = self.tab_clf.predict_proba(X_tab)
        text_proba = self.text_clf.predict_proba(X_text)
        proba = tab_proba + text_proba
        classes = self.tab_clf.classes_
        ids = np.argmax(proba, axis=1)
        predictions = [classes[i] for i in ids]
        return predictions


df = pd.read_csv('recipes.csv')
# print(df.isna().sum())
df = df.fillna(0)
df.loc[df.skill_level != 'Easy', 'skill_level'] = 'Not easy'
X, y = df.drop('skill_level', axis=1), df.skill_level
X_tab, X_text = X.drop('ingredients', axis=1), X.ingredients
del df

clf = make_pipeline(
    TfidfVectorizer(),
    FunctionTransformer(lambda x: x.todense()),
    BalancedRandomForestClassifier(n_jobs=-1, random_state=1)
)
score = cross_val_score(clf, X_text, y, scoring='balanced_accuracy').mean()
print(f"[Text] Balanced accuracy:{score:f}")

clf = BalancedRandomForestClassifier(n_jobs=1, random_state=1)
score = cross_val_score(clf, X_tab, y, scoring='balanced_accuracy').mean()
print(f"[Tabular] Balanced accuracy:{score:f}")

clf = CustomClassifier()
score = cross_val_score(clf, X, y, scoring='balanced_accuracy').mean()
print(f"[Both] Balanced accuracy:{score:f}")
