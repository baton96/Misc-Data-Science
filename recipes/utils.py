import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor

_df = pd.read_csv('recipes.csv')
# print(df.isna().sum())
columns = ['carbohydrate', 'kcal', 'protein', 'salt', 'saturatedfat', 'fat', 'sugar']


def autocorr(df):
    df = df[columns]
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    plt.show()


def ratio(df, col1, col2):
    df = df[df[col1].notna() & df[col2].notna()]
    print((df[col1] / df[col2]).mean())


def autoregressor(df):
    regressors = {}
    df = df.drop(['skill_level', 'ingredients'], axis=1)
    for col in columns:
        print(col)
        df2 = df[df[col].notna()]
        X, y = df2.drop(col, axis=1), df2[col]
        reg = HistGradientBoostingRegressor()
        reg.fit(X, y)
        regressors[col] = reg
    print()
    df = pd.read_csv('recipes.csv')
    for col in columns:
        print(col)
        X = df.drop(['skill_level', 'ingredients'], axis=1).drop(col, axis=1)
        predictions = pd.Series(regressors[col].predict(X))
        df[col] = df[col].fillna(predictions)
    df.to_csv('recipes2.csv')
