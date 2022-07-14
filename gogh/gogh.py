import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin

    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100
    v = cmax * 100
    return h, s, v


df = pd.read_csv('gogh.csv')
df.loc[df.Style != 'Realism', 'Style'] = 'Impressionism'
for col in ['Year', 'Genre']:
    pass  # print(df[[col, 'Style']].value_counts().sort_index())
X, y = df.drop(['Name', 'Style'], axis=1), df.Style
X.Genre = LabelEncoder().fit_transform(X.Genre)
X['r/b'] = X.r / X.b
X['r/g'] = X.r / X.g
X['g/b'] = X.g / X.b
'''
X['r-b'] = (X.r - X.b).abs()
X['r-g'] = (X.r - X.g).abs()
X['g-b'] = (X.g - X.b).abs()

hsv = [rgb_to_hsv(row.r, row.g, row.b) for row in df.itertuples()]
h, s, v = zip(*hsv)
X['h'] = h
X['s'] = s
X['v'] = v
'''
clf = BalancedRandomForestClassifier(n_jobs=-1, random_state=1)  # 0.893070
print("Cross-validation accuracy:%f" % cross_val_score(clf, X, y, scoring='balanced_accuracy').mean())

clf = LGBMClassifier(is_unbalance=True)  # 0.887598
print("Cross-validation accuracy:%f" % cross_val_score(clf, X, y, scoring='balanced_accuracy').mean())
