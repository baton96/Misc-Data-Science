import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('pokemon.csv')
X, y = df.drop(['name', 'legendary'], axis=1), LabelEncoder().fit_transform(df.legendary)
X = pd.get_dummies(X, columns=['type1', 'type2'])

nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
X2 = nca.fit_transform(X, y)
plt.scatter(X2[:, 0], X2[:, 1], c=y)
plt.show()