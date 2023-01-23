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

df = pd.read_csv('ingredients.csv')
# print(df.cuisine.value_counts())
X, y = df.ingredients, df.cuisine
del df

pipeline = make_pipeline(
    # TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), sublinear_tf=True),  # 0.790919
    TfidfVectorizer(use_idf=False),  # 0.789335
    FunctionTransformer(lambda x: x.toarray()),
    LinearSVC(dual=False),
)
print("Cross-validation accuracy:%f" % cross_val_score(pipeline, X, y).mean())
