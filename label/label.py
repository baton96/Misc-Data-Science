import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, RobustScaler
from sklearn.semi_supervised import LabelPropagation


def main(dataset_name='iris'):
    print('Dataset:', dataset_name)
    if dataset_name == 'iris':
        digits = datasets.load_iris()
        X = digits.data
        y = digits.target

        pipeline = make_pipeline(
            Normalizer(),
            LabelPropagation(kernel='knn', n_neighbors=7),
        )
    elif dataset_name == 'penguins':
        df = pd.read_csv('penguins.csv').dropna()
        X = df.drop(columns=['species', 'island'])
        X['sex'] = X['sex'] == 'MALE'
        y = LabelEncoder().fit_transform(df.species)

        pipeline = make_pipeline(
            RobustScaler(),
            LabelPropagation(kernel="knn", n_neighbors=5),
        )
    else:
        raise Exception('Pass dataset_name')
    indices = np.random.RandomState(1).rand(len(X)) < 0.9
    y_train = np.copy(y)
    y_train[indices] = -1

    pipeline.fit(X, y_train)
    predicted_labels = pipeline[-1].transduction_[indices]
    true_labels = y[indices]
    accuracy = sum(i == j for i, j in zip(predicted_labels, true_labels)) / len(true_labels)
    print('Accuracy:', accuracy)
    cm = confusion_matrix(true_labels, predicted_labels, labels=pipeline[-1].classes_)
    print(
        "Data: %d labeled & %d unlabeled (%d total)"
        % (len(y) - sum(indices), sum(indices), len(y))
    )
    print("Confusion matrix")
    print(cm)


main('iris')
print('-' * 20)
main('penguins')
