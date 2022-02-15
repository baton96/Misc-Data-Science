import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
warnings.filterwarnings("ignore")
np.random.seed(1)

hue_names = [
    'red',
    'blue',
    'yellow',
    'orange',
    'green',
]

df = pd.read_csv('nodes.csv')
#print(df.Temperature.value_counts())
#print(df.Hue_Name.value_counts())
df = df[df.Hue_Name.isin(hue_names)]
df.loc[df.Hue_Name == 'turquoise green, bluish green', 'Hue_Name'] = 'green'
X, y = df.Label, df.Temperature
del df

pipeline = make_pipeline(
    TfidfVectorizer(analyzer='char', ngram_range=(1, 8)),
    FunctionTransformer(lambda x: x.todense()),
    LinearSVC(),
)
print("Cross-validation accuracy:%f" % cross_val_score(pipeline, X, y).mean())
