import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

np.random.seed(1)

df = pd.read_csv('author.csv')
X, y = df.text, df.author
del df

pipeline = make_pipeline(
    TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), sublinear_tf=True),  # 0.846
    FunctionTransformer(lambda x: x.todense()),
    LinearSVC(),
)
print("Cross-validation accuracy:%f" % cross_val_score(pipeline, X, y).mean())
